# 基础数据结构

公共头文件位于 `include/DataStructure/`，命名空间为 `thermofem`。链接 `ThermoFEM::thermofem` 即可使用。

## 内存策略与所有权

`CudaBuffer<T>` 支持以下策略，并以 `MemoryStrategy`（`std::variant`）保存在对象中。
分配和释放通过受约束的策略访问器静态分派，不使用虚函数。

| 策略 | 分配方式 | 主机访问 | 设备访问 |
| --- | --- | --- | --- |
| `GlobalMemory` | `cudaMalloc` | 使用传输接口 | `deviceData()` |
| `PinnedMemory` | `cudaMallocHost` | `hostData()` | 不提供保证，请使用 mapped 策略 |
| `MappedMemory` | `cudaHostAllocMapped` | `hostData()` | `deviceData()` 返回设备别名 |
| `ManagedMemory` | `cudaMallocManaged` | `hostData()` | `deviceData()` |

默认策略为 global。所有非空分配需要可用的 CUDA 设备；映射和统一内存还需要对应设备能力。
元素类型受 `BufferElement` 约束：可平凡复制、默认构造及析构，不带 const/volatile，
不是指针类型，对齐不超过 `std::max_align_t`。不在 CUDA 分配上运行元素构造或析构，
新分配数据未初始化。缓冲区不适用于 `std::string` 等管理外部资源的类型。

- 复制构造、复制赋值：深拷贝，目标采用来源的内存策略和大小；修改来源不影响目标。
- 移动构造、移动赋值：接管来源分配、大小、所属设备和策略；来源为空并可再次使用。
- `copyDataFrom(source)`：只复制数据，不改变目标策略、大小及地址；要求大小相同，支持全部内存类型组合。
- `changeMemoryType(strategy)`：以新策略重新分配、复制内容并替换原存储；相同策略为空操作。
  该接口保留对象身份与数据，但通常改变底层地址，不表示在原物理地址上转换内存属性。
- `copyFromHost(span)` / `copyToHost(span)`：同步导入或导出全部元素，数组长度必须匹配。
- 空存储不调用 CUDA，地址为空；仍保留策略并参与复制、移动和转换。

复制赋值与类型转换具有强异常保证：分配或复制失败时原对象保持不变。
CUDA 错误抛出含原始错误码的 `CudaError`；尺寸、结构或可访问性错误抛出标准异常。
析构不抛异常并尽力释放资源，运行时关闭或设备丢失时无法保证 CUDA 释放操作成功。

```cpp
#include <DataStructure/CudaBuffer.cuh>
#include <array>

const std::array<double, 3> input{1.0, 2.0, 3.0};
thermofem::CudaBuffer<double> source(input, thermofem::PinnedMemory{});
thermofem::CudaBuffer<double> target(3, thermofem::GlobalMemory{});
target.copyDataFrom(source);  // target 仍为 global，原地址保持不变
target = source;             // target 变为 pinned，数据深拷贝
target.changeMemoryType(thermofem::MappedMemory{});
auto* devicePointer = target.deviceData();
```

## 设备与同步边界

非空存储记录创建时的 CUDA 设备，可通过 `deviceId()` 查询。
复制、传输、类型转换和释放时自动切换至所属设备，并在退出时恢复调用方设备。
深拷贝始终在来源设备分配；`copyDataFrom` 明确拒绝不同设备的非空分配，尚不提供 peer copy。
发射内核之前由调用方选择存储所属设备。

当前接口不提供 stream 或异步传输。调用方须在主机读写、数据复制、转换、覆盖及销毁前，
同步所有可能访问这些存储的外部内核或异步传输。尤其不能在 GPU 写 mapped/managed 内存时
直接在 CPU 上读写。`data()` 只提供原始分配地址，不保证它适用于两端；
使用 `hostData()` / `deviceData()` 表达访问意图。转换、赋值、移动或对象销毁后需重新获取指针。

## 向量与稠密矩阵

`Vector<T>` 通过组合独占一个 `CudaBuffer<T>`，二者没有继承关系，也没有隐式转换。
向量长度直接取自存储，不维护重复的尺寸状态。复制与移动采用来源策略，
仅数据复制保留目标策略及地址，转换策略具有强异常保证。
常用存储操作由向量显式转发；跨存储抽象使用 `vector.values()`，该接口只读，
避免外部替换向量内部的 Buffer。原先依赖 Vector 自动绑定 Buffer 引用的代码须改为显式访问。
数值运算属于 `LinearAlgebra` 层，不放入存储类。

`DenseMatrix<T>` 使用行主序，元素 `(row, column)` 的偏移为 `row * columnCount() + column`。
构造时检查维度乘积与字节数是否溢出；`at(row, column)` 检查边界及主机可访问性。
`values()` 返回只读存储接口，保护元素数与维度一致；可通过传输接口和 `deviceData()` 修改数据。
`copyDataFrom` 还要求形状完全一致。复制赋值覆盖形状、数据及策略，移动后来源为 0 行 0 列。

```cpp
#include <DataStructure/DenseMatrix.cuh>

thermofem::DenseMatrix<double> matrix(2, 3, thermofem::PinnedMemory{});
matrix.at(1, 2) = 7.0;
matrix.changeMemoryType(thermofem::GlobalMemory{});
```

## CSR 稀疏矩阵

`CsrMatrix<T, Index>` 保存行数、列数和三个独立且策略一致的缓冲区：
`rowOffsets()`、`columnIndices()` 和 `values()`。`Index` 默认为 `std::int32_t`，
受 `CsrIndex` 约束，支持 32 位或 64 位有符号整数。

从主机数组构造时，在任何 CUDA 分配前检查：

- 行、列、非零数可由索引类型表示。
- 行偏移长度为行数加一，列索引数与非零元素数一致。
- 首个行偏移为 0，末项为非零数，中间项单调不减且位于合法区间。
- 每个列索引非负且严格小于列数。

允许空行、未排序的列索引、重复列索引和显式零值；当前不自动排序、合并或消除零元素。
`nonZeroCount()` 是存储条目数。默认及移动后的空对象不分配行偏移；
显式构造的零行矩阵仍需提供单项行偏移 `{0}`。

结构数组仅提供只读接口。`copyValuesFromHost` 只更新数值，`deviceValues()` 用于内核修改数值。
复制及移动作用于整个矩阵；`changeMemoryType` 转换三个缓冲区并保留结构及数值，
失败时保持原矩阵，过程中需要额外临时存储。

```cpp
#include <DataStructure/CsrMatrix.cuh>

const int offsets[]{0, 1, 2};
const int columns[]{0, 1};
const double values[]{2.0, 3.0};
thermofem::CsrMatrix<double> matrix(2, 2, offsets, columns, values);
```

## 验证

`test/data_structure_test.cpp` 验证内存类型传播、16 种源目标组合的数据复制与转换、
移动及自赋值、空对象、尺寸溢出、主机数组长度、稠密布局与 CSR 合法性。
没有可用设备时，真实 CUDA 用例明确标记跳过，主机校验用例仍运行。

`example-2` 在 global、mapped 和 managed 内存上实际发射 CUDA 内核，
核对计算结果，并演示稠密矩阵和 CSR 存储。运行需要设备支持这三类内存：

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/example/example-2
```

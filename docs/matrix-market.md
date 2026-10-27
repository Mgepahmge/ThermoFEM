# Matrix Market 用户手册

## 输入格式

`<IO/MatrixMarket.cuh>` 提供 `thermofem::io::readMatrixMarket`，从输入流读取
Matrix Market 坐标格式，返回主机端 `HostCsr`，供数据结构和求解组件使用。

支持实数或整数条目，以及 general 和 symmetric 矩阵。读取时检查文件头、
维度、条目数量、索引范围和数值有限性；对称存储按其格式展开。
重复坐标可保留在主机 CSR 中，设备计算接口负责所需的规范化。

```cpp
#include <IO/MatrixMarket.cuh>
#include <DataStructure/CsrMatrix.cuh>
#include <fstream>

std::ifstream input("data/matrix.mtx");
if (!input) {
    throw std::runtime_error("Cannot open matrix file");
}
const auto host = thermofem::io::readMatrixMarket(input);
thermofem::CsrMatrix<double> matrix(
    host.rows_, host.columns_, host.offsets_, host.indices_, host.values_);
```

线性求解要求非空方阵。AMG 还要求各层对角非零且倒数有限，原生聚合结构受
32 位索引容量限制；具体配置和异常语义见 [求解组件设计](solver.md)。

## 命令行示例

启用 `THERMOFEM_BUILD_EXAMPLES` 时构建 `solve-matrix`。在项目根目录执行：

```bash
./build/example/solve-matrix data/matrix.mtx amg v
./build/example/solve-matrix data/matrix.mtx amg w
./build/example/solve-matrix data/matrix.mtx amg f
./build/example/solve-matrix data/matrix.mtx identity
```

参数依次为文件路径、预条件子和循环类型。预条件子默认为 `amg`，循环默认为 `v`；
`identity` 表示恒等预条件子，此时循环参数不参与计算。
数据文件由使用者放入 `data/`，其内容不纳入 Git。

此程序使用确定性的制造解检验线性求解，并不读取原工况的右端项：

```text
xExact[i] = 1 + 0.25*sin(0.013*i) + 0.125*cos(0.007*i)
b = A*xExact
xInitial = 0
```

计算采用 double 精度，相对容差为 `1e-8`，绝对容差为零，最大迭代数为 1000，
每 50 次迭代刷新真实残差。右端由主机端扩展精度累加得到，求解后独立重算残差和解误差。
读取实际右端项并求解业务问题时，应通过 `BiCgStab::setup/apply` 接口传入数据。

## 输出与退出码

程序输出矩阵规模、设备信息、读取及求解耗时、终止状态、迭代数、相对残差与解误差。
AMG 模式还输出循环类型、停止粗化的原因、粗层求解方式，以及每层规模和聚合统计。

算子复杂度是所有层非零数之和除以最细层非零数；网格复杂度是所有层行数之和
除以最细层行数。耗时不包含独立主机误差校验，单次程序运行不构成稳定性能基准。

| 退出码 | 含义 |
| --- | --- |
| 0 | 求解报告收敛，主机相对残差不超过 `1.01e-8`，相对解误差不超过 `1e-4` |
| 1 | 文件、参数、矩阵或计算过程发生异常 |
| 2 | 求解完成，但未满足上述数值校验条件 |

制造解校验用于判断该矩阵上的线性求解结果，不代表原始工况或物理模型的正确性。

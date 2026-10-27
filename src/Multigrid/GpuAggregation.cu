/** @file GpuAggregation.cu
 * @brief 实现 GPU 两阶段匹配、图收缩、单点吸收、直接 Galerkin 归并、传输与粗层 LU。
 * @details 数值算法参考用户的非公开 YHAMG-GPU，保留匹配哈希与 GPU 阶段顺序。
 */
#include <Multigrid/GpuAggregation.cuh>
#include <cub/cub.cuh>
#include <climits>
#include <cmath>
#include <concepts>
#include <functional>
namespace thermofem::multigrid {
/** @brief 封装仅依赖 GPU 数组的 setup 与传输内核。 */
namespace {
using Key = unsigned long long;
using Matrix = CsrMatrix<double>;
/** @brief CUB 求和归并算子。@code Sum{}(1,2); @endcode */
struct Sum {
    __host__ __device__ double operator()(double a, double b) const {
        return a + b;
    }
};
/** @brief CUB 最大权归并算子。@code Maximum{}(1,2); @endcode */
struct Maximum {
    __host__ __device__ double operator()(double a, double b) const {
        return a > b ? a : b;
    }
};
constexpr int block = 256;
/** @brief 计算线程块数。@param n 线程数。@pre n>0。@return 块数。@code grid(n); @endcode */
int grid(int n) {
    return (n + size_t(block) - 1) / block;
}
/** @brief 检查 CUDA 错误。@param status 返回码。@param operation 操作名。
 * @pre 无额外约束。@return 无返回值。@code checkCuda(cudaGetLastError(),"kernel"); @endcode */
void checkCuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        throw CudaError(status, operation);
}
/** @brief 将行列拼为键。@param r 行。@param c 列。@pre 非负。@return 键。@code key(1,2); @endcode */
__device__ Key key(int r, int c) {
    return (Key(unsigned(r)) << 32) | unsigned(c);
}
/** @brief 计算列号所需位数。@param rows 维度。@pre 正维度。@return 位数。@code columnBits(64); @endcode */
int columnBits(int rows) {
    int bits = 1;
    for (unsigned r = unsigned(rows - 1); r >>= 1;)
        ++bits;
    return bits;
}
/** @brief 压缩坐标键。@param r 行。@param c 列。@param bits 列位宽。
 * @pre 位宽可容纳列。@return 键。@code sortKey(1,2,8); @endcode */
__device__ Key sortKey(int r, int c, int bits) {
    return (Key(unsigned(r)) << bits) | unsigned(c);
}
/** @brief 对称边哈希，用于匹配强度相同时的确定性选择。
 * @param i 端点。@param j 端点。@pre 非负。@return 优先级。@code edgePriority(1,2); @endcode */
__device__ Key edgePriority(int i, int j) {
    Key h = key(min(i, j), max(i, j));
    h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
    h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
    return h ^ (h >> 31);
}
/** @brief 组合 global 缓冲区，保留最大工作容量；不承载矩阵语义。
 * @code WorkspaceBuffer<int> buffer(10); buffer.reserve(20); @endcode */
template <BufferElement T> struct WorkspaceBuffer {
    CudaBuffer<T> buffer_; ///< 工作存储及其生命周期。
    /** @brief 分配容量。@param n 容量。@pre 当前设备有效。@return 无返回值。@code WorkspaceBuffer<int> b(4);
     * @endcode */
    explicit WorkspaceBuffer(size_t n = 0) : buffer_(n) {}
    /** @brief 按需增长。@param n 所需容量。@pre 无异步访问。@return 无返回值。@code b.reserve(8); @endcode */
    void reserve(size_t n) {
        if (n > buffer_.size())
            buffer_ = CudaBuffer<T>(n);
    }
    /** @brief 获取设备地址，无参数、无额外约束。@return 地址。@code b.data(); @endcode */
    T *data() {
        return buffer_.data();
    }
    /** @brief 获取只读地址，无参数、无额外约束。@return 地址。@code b.data(); @endcode */
    const T *data() const {
        return buffer_.data();
    }
    /** @brief 查询容量字节数，无参数、无额外约束。@return 字节数。@code b.bytes(); @endcode */
    size_t bytes() const {
        return buffer_.sizeBytes();
    }
};
/** @brief 组合排序、归并、匹配、成员表及扫描缓存，跨层复用最大容量。
 * @details 仅成员临时快照在吸收时与最终映射交换所有权；不持有矩阵或层次。
 * @code Workspace workspace(false); @endcode */
struct Workspace {
    WorkspaceBuffer<Key> inputKeys_, sortedKeys_, uniqueKeys_; ///< 输入、排序、唯一坐标键。
    WorkspaceBuffer<double> inputValues_, sortedValues_, uniqueValues_,
        maximum_;                                                   ///< 对应数值和各行最大非对角强度。
    WorkspaceBuffer<int> count_, mate_, proposal_, flags_, prefix_; ///< 标量、匹配、提议、扫描标志和前缀。
    WorkspaceBuffer<int> sizes_, memberSlots_, nextMap_;            ///< 成员数、固定四槽成员、吸收快照。
    WorkspaceBuffer<double> proposalWeight_;                        ///< 单点提议强度。
    WorkspaceBuffer<unsigned char> scratch_;                        ///< CUB 临时存储。
    bool compact_;                                                  ///< 是否使用紧凑排序键。
    /** @brief 创建复用缓存。@param compactKeys 紧凑键开关。@pre 当前设备有效。
     * @return 无返回值。@code Workspace workspace(true); @endcode */
    explicit Workspace(bool compactKeys = false) : count_(1), compact_(compactKeys) {}
    /** @brief 查询键的列位宽。@param rows 阶数。@pre 正数。@return 位宽。@code workspace.bits(64); @endcode
     */
    int bits(int rows) const {
        return compact_ ? columnBits(rows) : 32;
    }
    /** @brief 增长坐标缓冲区。@param n 条目数。@pre 非负。@return 无返回值。@code workspace.entries(10);
     * @endcode */
    void entries(int n) {
        inputKeys_.reserve(n);
        sortedKeys_.reserve(n);
        inputValues_.reserve(n);
        sortedValues_.reserve(n);
        if (!compact_) {
            uniqueKeys_.reserve(n);
            uniqueValues_.reserve(n);
        }
    }
    /** @brief 回传设备标量，无参数。@pre count 已写入。@return 标量。@code workspace.scalar(); @endcode */
    int scalar() {
        int result;
        checkCuda(cudaMemcpy(&result, count_.data(), sizeof(int), cudaMemcpyDeviceToHost),
                  "AMG scalar download");
        return result;
    }
    /** @brief 复用临时容量执行 GPU 排他前缀和。
     * @param in 输入。@param out 输出。@param n 长度。@pre 数组有效且不别名。
     * @return 无返回值。@code workspace.scan(flags,prefix,n); @endcode */
    void scan(int *in, int *out, int n) {
        size_t bytes = 0;
        checkCuda(cub::DeviceScan::ExclusiveSum(nullptr, bytes, in, out, n), "AMG scan size");
        scratch_.reserve(bytes);
        checkCuda(cub::DeviceScan::ExclusiveSum(scratch_.data(), bytes, in, out, n), "AMG scan");
    }
};

/** @brief 逐条映射矩阵坐标，为规范化或直接 Galerkin 生成键值数组。
 * @param n 本阶段维度或条目数。
 * @param rowOffsets 当前设备上的rowOffsets数组或参数。
 * @param columnIndices 当前设备上的columnIndices数组或参数。
 * @param entries 当前设备上的entries数组或参数。
 * @param map 当前设备上的map数组或参数。
 * @param keys 当前设备上的keys数组或参数。
 * @param values 当前设备上的values数组或参数。
 * @param bits 当前设备上的bits数组或参数。
 * @pre 上一阶段的结构与维度有效，当前设备正确，各数组容量满足对应阶段要求。
 * @return 无返回值。
 * @code mapEntries<<<blocks,256>>>(n,rowOffsets,columnIndices,entries,map,keys,values,bits); @endcode */
__global__ void mapEntries(int n, const int *rowOffsets, const int *columnIndices, const double *entries,
                           const int *map, Key *keys, double *values, int bits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    for (int j = rowOffsets[i]; j < rowOffsets[i + 1]; ++j) {
        keys[j] = sortKey(map ? map[i] : i, map ? map[columnIndices[j]] : columnIndices[j], bits);
        values[j] = entries[j];
    }
}
/** @brief 解包唯一坐标并统计每行条目数。
 * @param nnz 输入条目数。
 * @param keys 排序坐标键。
 * @param v 归并后的数值。
 * @param counts 行条目计数。
 * @param columnIndices CSR 列索引。
 * @param entries CSR 系数。
 * @param bits 键的列位宽。
 * @pre 当前设备正确，数组有效，维度与本阶段匹配，结构由上一阶段验证。
 * @return 无返回值。
 * @details 行计数供后续排他扫描生成 CSR 偏移，不改变已排序条目顺序。
 * @code unpack(nnz,keys,v,counts,columnIndices,entries,bits); @endcode */
__global__ void unpack(int nnz, const Key *keys, const double *v, int *counts, int *columnIndices,
                       double *entries, int bits) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nnz)
        return;
    atomicAdd(counts + int(keys[j] >> bits), 1);
    columnIndices[j] = int(keys[j] & ((Key(1) << bits) - 1));
    entries[j] = v[j];
}

/** @brief CUB 坐标排序与重复键归并，供规范化、强图和 Galerkin 复用。
 * @param workspace 复用存储，输入位于 inputKeys_/inputValues_。
 * @param rows 输出阶数。@param nnz 输入条目数。@param reduce 求和或最大权归并。
 * @pre 坐标非负且小于 rows，nnz 可由 int 表示，当前设备正确。
 * @return 行列有序、无重复坐标的 CSR，保留归并得到的显式零。
 * @details 排序和归并使用互不重叠的输入输出。紧凑键可复用排序前的输入存储。
 * 唯一项计数回传后分配准确大小的矩阵，再统计行数、前缀扫描生成行偏移。
 * @code auto coarse=compress(workspace,rows,nnz,Sum{}); @endcode */
template <class Reduce>
    requires std::invocable<Reduce, double, double>
Matrix compress(Workspace &workspace, int rows, int nnz, Reduce reduce) {
    if (nnz == 0) {
        CudaBuffer<int> rowOffsets(size_t(rows) + 1), columnIndices;
        CudaBuffer<double> entries;
        checkCuda(cudaMemset(rowOffsets.data(), 0, (size_t(rows) + 1) * sizeof(int)), "AMG empty graph");
        return Matrix::fromBuffers(rows, rows, std::move(rowOffsets), std::move(columnIndices),
                                   std::move(entries));
    }
    const int bits = workspace.bits(rows), endBit = bits + columnBits(rows);
    size_t bytes = 0;
    checkCuda(cub::DeviceRadixSort::SortPairs(nullptr, bytes, workspace.inputKeys_.data(),
                                              workspace.sortedKeys_.data(), workspace.inputValues_.data(),
                                              workspace.sortedValues_.data(), nnz, 0, endBit),
              "AMG sort size");
    workspace.scratch_.reserve(bytes);
    checkCuda(cub::DeviceRadixSort::SortPairs(workspace.scratch_.data(), bytes, workspace.inputKeys_.data(),
                                              workspace.sortedKeys_.data(), workspace.inputValues_.data(),
                                              workspace.sortedValues_.data(), nnz, 0, endBit),
              "AMG sort");
    bytes = 0;

    Key *uniqueKeys = workspace.compact_ ? workspace.inputKeys_.data() : workspace.uniqueKeys_.data();
    double *uniqueValues =
        workspace.compact_ ? workspace.inputValues_.data() : workspace.uniqueValues_.data();
    checkCuda(cub::DeviceReduce::ReduceByKey(nullptr, bytes, workspace.sortedKeys_.data(), uniqueKeys,
                                             workspace.sortedValues_.data(), uniqueValues,
                                             workspace.count_.data(), reduce, nnz),
              "AMG reduce size");
    workspace.scratch_.reserve(bytes);
    checkCuda(cub::DeviceReduce::ReduceByKey(workspace.scratch_.data(), bytes, workspace.sortedKeys_.data(),
                                             uniqueKeys, workspace.sortedValues_.data(), uniqueValues,
                                             workspace.count_.data(), reduce, nnz),
              "AMG reduce");
    int unique = workspace.scalar();
    CudaBuffer<int> rowOffsets(size_t(rows) + 1), columnIndices(unique);
    CudaBuffer<double> entries(unique);
    workspace.flags_.reserve(size_t(rows) + 1);
    checkCuda(cudaMemset(workspace.flags_.data(), 0, (size_t(rows) + 1) * sizeof(int)), "AMG row counts");
    unpack<<<grid(unique), block>>>(unique, uniqueKeys, uniqueValues, workspace.flags_.data(),
                                    columnIndices.data(), entries.data(), bits);
    workspace.scan(workspace.flags_.data(), rowOffsets.data(), rows + 1);
    return Matrix::fromBuffers(rows, rows, std::move(rowOffsets), std::move(columnIndices),
                               std::move(entries));
}

/** @brief 将细条目映射到粗坐标后归并，不产生稀疏乘法中间结果。
 * @param a 细矩阵。@param map 细到粗映射，空表示规范化原坐标。
 * @param rows 输出阶数。@param workspace 工作区。@pre 有效映射、当前设备正确。
 * @return 粗矩阵，条目数不超过输入。@code auto ac=galerkin(a,map,rows,workspace); @endcode */
Matrix galerkin(const Matrix &a, const int *map, int rows, Workspace &workspace) {
    int nnz = int(a.nonZeroCount());
    workspace.entries(nnz);
    mapEntries<<<grid(int(a.rowCount())), block>>>(int(a.rowCount()), a.rowOffsets().deviceData(),
                                                   a.columnIndices().deviceData(), a.values().deviceData(),
                                                   map, workspace.inputKeys_.data(),
                                                   workspace.inputValues_.data(), workspace.bits(rows));
    return compress(workspace, rows, nnz, Sum());
}

/** @brief 校验本层数值与对角倒数，同时计算行最大非对角强度。
 * @param n 本阶段工作维度。
 * @param rowOffsets CSR 行偏移。
 * @param columnIndices CSR 列索引。
 * @param a 输入矩阵或系数。
 * @param bad 错误标志。
 * @param maximum 行最大非对角强度。
 * @pre 当前设备正确，数组有效，维度与本阶段匹配，结构由上一阶段验证。
 * @return 无返回值。
 * @details 每行独立扫描，错误通过整数标志归并；最大强度缓存在 setup 工作区。
 * @code validate(n,rowOffsets,columnIndices,a,bad,maximum); @endcode */
__global__ void validate(int n, const int *rowOffsets, const int *columnIndices, const double *a, int *bad,
                         double *maximum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    double diag = 0, mx = 0;
    for (int j = rowOffsets[i]; j < rowOffsets[i + 1]; ++j) {
        if (!isfinite(a[j]))
            atomicExch(bad, 1);
        if (columnIndices[j] == i)
            diag = a[j];
        else
            mx = fmax(mx, fabs(a[j]));
    }
    if (diag == 0 || !isfinite(diag) || !isfinite(1.0 / diag))
        atomicExch(bad, 1);
    if (maximum)
        maximum[i] = mx;
}
/** @brief 检查 CSR 边界、索引、有限值及列有序唯一性。
 * @param n 本阶段工作维度。
 * @param nnz 输入条目数。
 * @param rowOffsets CSR 行偏移。
 * @param columnIndices CSR 列索引。
 * @param entries CSR 系数。
 * @param flags 扫描或校验标志。
 * @pre 当前设备正确，数组有效，维度与本阶段匹配，结构由上一阶段验证。
 * @return 无返回值。
 * @details 标志位 1 表示需要排序归并，位 2 表示非法结构或非有限系数。
 * @code checkCanonical(n,nnz,rowOffsets,columnIndices,entries,flags); @endcode */
__global__ void checkCanonical(int n, int nnz, const int *rowOffsets, const int *columnIndices,
                               const double *entries, int *flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    if (rowOffsets[0] != 0 || rowOffsets[n] != nnz || rowOffsets[i] < 0 ||
        rowOffsets[i] > rowOffsets[i + 1] || rowOffsets[i + 1] > nnz) {
        atomicOr(flags, 2);
        return;
    }
    int previous = -1;
    for (int j = rowOffsets[i]; j < rowOffsets[i + 1]; ++j) {
        if (columnIndices[j] < 0 || columnIndices[j] >= n || !isfinite(entries[j]))
            atomicOr(flags, 2);
        if (columnIndices[j] <= previous)
            atomicOr(flags, 1);
        previous = columnIndices[j];
    }
}
/** @brief 校验层矩阵并保存行最大非对角强度。
 * @param a 输入矩阵或系数。
 * @param workspace 复用 GPU 工作区。
 * @pre 当前设备正确，数组有效，维度与本阶段匹配，结构由上一阶段验证。
 * @return 无返回值。
 * @details 仅读回标量错误标志；对角缺失、零或倒数非有限均拒绝。
 * @code validateMatrix(a,workspace); @endcode */
void validateMatrix(const Matrix &a, Workspace &workspace) {
    checkCuda(cudaMemset(workspace.count_.data(), 0, sizeof(int)), "AMG validation flag");
    workspace.maximum_.reserve(a.rowCount());
    validate<<<grid(int(a.rowCount())), block>>>(int(a.rowCount()), a.rowOffsets().deviceData(),
                                                 a.columnIndices().deviceData(), a.values().deviceData(),
                                                 workspace.count_.data(), workspace.maximum_.data());
    if (workspace.scalar())
        throw std::invalid_argument(
            "Native AMG: missing/zero diagonal, non-finite value or reciprocal in hierarchy");
}
/** @brief 计数或填充双向强边；空键指针表示计数，否则使用扫描后的行写入偏移。
 * @param n 本阶段维度或条目数。
 * @param rowOffsets 当前设备上的rowOffsets数组或参数。
 * @param columnIndices 当前设备上的columnIndices数组或参数。
 * @param entries 当前设备上的entries数组或参数。
 * @param maximum 当前设备上的maximum数组或参数。
 * @param threshold 当前设备上的threshold数组或参数。
 * @param keys 当前设备上的keys数组或参数。
 * @param weights 当前设备上的weights数组或参数。
 * @param count 当前设备上的count数组或参数。
 * @param bits 当前设备上的bits数组或参数。
 * @pre 上一阶段的结构与维度有效，当前设备正确，各数组容量满足对应阶段要求。
 * @return 无返回值。
 * @code
 * strength<<<blocks,256>>>(n,rowOffsets,columnIndices,entries,maximum,threshold,keys,weights,count,bits);
 * @endcode */
__global__ void strength(int n, const int *rowOffsets, const int *columnIndices, const double *entries,
                         const double *maximum, double threshold, Key *keys, double *weights, int *count,
                         int bits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int pos = keys ? count[i] : 0;
    for (int j = rowOffsets[i]; j < rowOffsets[i + 1]; ++j) {
        int c = columnIndices[j];
        double weight = (i != c && maximum[i] > 0 && fabs(entries[j]) / maximum[i] >= threshold)
                            ? fabs(entries[j]) / maximum[i]
                            : 0;
        if (weight > 0) {
            if (keys) {
                keys[pos] = sortKey(i, c, bits);
                keys[pos + 1] = sortKey(c, i, bits);
                weights[pos] = weights[pos + 1] = weight;
            }
            pos += 2;
        }
    }
    if (!keys)
        count[i] = pos;
}
/** @brief 构造对称强图，先计数扫描，再填充双向边，最后最大权归并。
 * @param a 规范矩阵。@param threshold 强度阈值。@param workspace 已缓存行最大强度的工作区。
 * @pre 已对同一矩阵调用 validateMatrix。@return 无自环的对称正权图。
 * @code auto graph=strengthGraph(a,.25,workspace); @endcode */
Matrix strengthGraph(const Matrix &a, double threshold, Workspace &workspace) {
    if (a.nonZeroCount() > size_t(INT_MAX / 2))
        throw std::overflow_error("Native AMG: strength graph exceeds 32-bit capacity");
    int n = int(a.rowCount());
    workspace.flags_.reserve(size_t(n) + 1);
    workspace.prefix_.reserve(size_t(n) + 1);
    checkCuda(cudaMemset(workspace.flags_.data() + n, 0, sizeof(int)), "AMG strength counter");
    strength<<<grid(n), block>>>(n, a.rowOffsets().deviceData(), a.columnIndices().deviceData(),
                                 a.values().deviceData(), workspace.maximum_.data(), threshold, nullptr,
                                 nullptr, workspace.flags_.data(), workspace.bits(n));
    workspace.scan(workspace.flags_.data(), workspace.prefix_.data(), n + 1);
    int nnz = 0;
    checkCuda(cudaMemcpy(&nnz, workspace.prefix_.data() + n, sizeof(int), cudaMemcpyDeviceToHost),
              "AMG strong edge count");
    workspace.entries(nnz);
    strength<<<grid(n), block>>>(n, a.rowOffsets().deviceData(), a.columnIndices().deviceData(),
                                 a.values().deviceData(), workspace.maximum_.data(), threshold,
                                 workspace.inputKeys_.data(), workspace.inputValues_.data(),
                                 workspace.prefix_.data(), workspace.bits(n));
    return compress(workspace, n, nnz, Maximum());
}
/** @brief 基于固定匹配快照为每个未匹配点选择最优邻居。
 * @param n 本阶段工作维度。
 * @param rowOffsets CSR 行偏移。
 * @param columnIndices CSR 列索引。
 * @param weights 强度或排序候选权重。
 * @param mate 当前配对端点。
 * @param proposal 每个细点的提议。
 * @pre 当前设备正确，数组有效，维度与本阶段匹配，结构由上一阶段验证。
 * @return 无返回值。
 * @details 按权重、对称边哈希降序、邻点编号升序决定唯一提议；不在本内核更新匹配。
 * @code propose(n,rowOffsets,columnIndices,weights,mate,proposal); @endcode */
__global__ void propose(int n, const int *rowOffsets, const int *columnIndices, const double *weights,
                        const int *mate, int *proposal) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int best = -1;
    double score = 0;
    Key priority = 0;
    if (mate[i] < 0)
        for (int j = rowOffsets[i]; j < rowOffsets[i + 1]; ++j) {
            int c = columnIndices[j];
            double s = weights[j];
            if (c == i || mate[c] >= 0 || s <= 0)
                continue;
            if (s < score)
                continue;
            Key candidate = edgePriority(i, c);
            if (s > score ||
                (s == score && (best < 0 || candidate > priority || (candidate == priority && c < best)))) {
                best = c;
                score = s;
                priority = candidate;
            }
        }
    proposal[i] = best;
}
/** @brief 接受互相提议的端点对。
 * @param n 本阶段工作维度。
 * @param proposal 每个细点的提议。
 * @param mate 当前配对端点。
 * @pre 当前设备正确，数组有效，维度与本阶段匹配，结构由上一阶段验证。
 * @return 无返回值。
 * @details 全部提议已由上一内核写完；每个线程仅写本端点，避免读写快照混用。
 * @code accept(n,proposal,mate); @endcode */
__global__ void accept(int n, const int *proposal, int *mate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int p = proposal[i];
    if (p >= 0 && proposal[p] == i)
        mate[i] = p;
}
/** @brief 标记未匹配点或匹配对较小端点为根。
 * @param n 本阶段工作维度。
 * @param mate 当前配对端点。
 * @param flags 扫描或校验标志。
 * @pre 当前设备正确，数组有效，维度与本阶段匹配，结构由上一阶段验证。
 * @return 无返回值。
 * @details 额外尾哨兵为零，后续排他扫描同时产生连续编号和聚合总数。
 * @code roots(n,mate,flags); @endcode */
__global__ void roots(int n, const int *mate, int *flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n)
        flags[i] = (i < n && (mate[i] < 0 || i < mate[i])) ? 1 : 0;
}
/** @brief 用根前缀和生成每个细点的连续聚合编号。
 * @param n 本阶段工作维度。
 * @param mate 当前配对端点。
 * @param prefix 排他前缀和。
 * @param map 细到粗聚合映射，规范化时允许为空。
 * @pre 当前设备正确，数组有效，维度与本阶段匹配，结构由上一阶段验证。
 * @return 无返回值。
 * @details 配对两端读取相同根编号；未匹配点读取自身编号。
 * @code assign(n,mate,prefix,map); @endcode */
__global__ void assign(int n, const int *mate, const int *prefix, int *map) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        map[i] = prefix[mate[i] < 0 ? i : min(i, mate[i])];
}
/** @brief 同步互选匹配，每轮先完成全部提议，再接受相互提议的边。
 * @param graph 对称正权图。@param rounds 固定轮数。@param map 输出连续聚合编号。
 * @param workspace 工作区。@pre map 容量等于图阶数，当前设备正确。
 * @return 聚合数量。@details 未匹配点独立成组；根为较小端点，排他前缀和压缩编号。
 * 两个独立内核保证提议读取一致快照，不能将读写合并为顺序贪心。
 * @code auto count=pair(graph,8,map,workspace); @endcode */
int pair(const Matrix &graph, int rounds, int *map, Workspace &workspace) {
    int n = int(graph.rowCount());
    workspace.mate_.reserve(n);
    workspace.proposal_.reserve(n);
    workspace.flags_.reserve(size_t(n) + 1);
    workspace.prefix_.reserve(size_t(n) + 1);
    checkCuda(cudaMemset(workspace.mate_.data(), 0xff, size_t(n) * sizeof(int)), "AMG unmatched flags");
    for (int r = 0; r < rounds; ++r) {
        propose<<<grid(n), block>>>(n, graph.rowOffsets().deviceData(), graph.columnIndices().deviceData(),
                                    graph.values().deviceData(), workspace.mate_.data(),
                                    workspace.proposal_.data());
        accept<<<grid(n), block>>>(n, workspace.proposal_.data(), workspace.mate_.data());
    }
    roots<<<grid(n + 1), block>>>(n, workspace.mate_.data(), workspace.flags_.data());
    workspace.scan(workspace.flags_.data(), workspace.prefix_.data(), n + 1);
    assign<<<grid(n), block>>>(n, workspace.mate_.data(), workspace.prefix_.data(), map);
    int count;
    checkCuda(cudaMemcpy(&count, workspace.prefix_.data() + n, sizeof(int), cudaMemcpyDeviceToHost),
              "AMG aggregate count");
    return count;
}
/** @brief 原地组合细到中间层和中间到粗层的映射。
 * @param n 本阶段工作维度。
 * @param map 细到粗聚合映射，规范化时允许为空。
 * @param second 中间到粗层映射。
 * @pre 当前设备正确，数组有效，维度与本阶段匹配，结构由上一阶段验证。
 * @return 无返回值。
 * @details 每线程只改一个细点，第二映射只读且与输出不别名。
 * @code compose(n,map,second); @endcode */
__global__ void compose(int n, int *map, const int *second) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        map[i] = second[map[i]];
}
/** @brief 按第一阶段映射收缩图，过滤内部边，生成待最大权归并的跨聚合边。
 * @param n 本阶段维度或条目数。
 * @param rowOffsets 当前设备上的rowOffsets数组或参数。
 * @param columnIndices 当前设备上的columnIndices数组或参数。
 * @param entries 当前设备上的entries数组或参数。
 * @param map 当前设备上的map数组或参数。
 * @param offsets 当前设备上的offsets数组或参数。
 * @param keys 当前设备上的keys数组或参数。
 * @param values 当前设备上的values数组或参数。
 * @param bits 当前设备上的bits数组或参数。
 * @pre 上一阶段的结构与维度有效，当前设备正确，各数组容量满足对应阶段要求。
 * @return 无返回值。
 * @code mapGraph<<<blocks,256>>>(n,rowOffsets,columnIndices,entries,map,offsets,keys,values,bits); @endcode
 */
__global__ void mapGraph(int n, const int *rowOffsets, const int *columnIndices, const double *entries,
                         const int *map, int *offsets, Key *keys, double *values, int bits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int pos = keys ? offsets[i] : 0;
    for (int j = rowOffsets[i]; j < rowOffsets[i + 1]; ++j)
        if (map[i] != map[columnIndices[j]] && entries[j] > 0) {
            if (keys) {
                keys[pos] = sortKey(map[i], map[columnIndices[j]], bits);
                values[pos] = entries[j];
            }
            ++pos;
        }
    if (!keys)
        offsets[i] = pos;
}
/** @brief 按第一阶段聚合收缩强图，聚合内边删除，平行边取最大权。
 * @param g 强图。@param map 连续聚合编号。@param rows 粗阶数。@param workspace 工作区。
 * @pre 映射合法、当前设备正确。@return 第二阶段匹配的强图。
 * @code auto contracted=contractGraph(graph,map,rows,workspace); @endcode */
Matrix contractGraph(const Matrix &g, const int *map, int rows, Workspace &workspace) {
    int n = int(g.rowCount());
    workspace.flags_.reserve(size_t(n) + 1);
    workspace.prefix_.reserve(size_t(n) + 1);
    checkCuda(cudaMemset(workspace.flags_.data() + n, 0, sizeof(int)), "AMG graph count");
    mapGraph<<<grid(n), block>>>(n, g.rowOffsets().deviceData(), g.columnIndices().deviceData(),
                                 g.values().deviceData(), map, workspace.flags_.data(), nullptr, nullptr,
                                 workspace.bits(rows));
    workspace.scan(workspace.flags_.data(), workspace.prefix_.data(), n + 1);
    int nnz = 0;
    checkCuda(cudaMemcpy(&nnz, workspace.prefix_.data() + n, sizeof(int), cudaMemcpyDeviceToHost),
              "AMG contracted edge count");
    workspace.entries(nnz);
    mapGraph<<<grid(n), block>>>(n, g.rowOffsets().deviceData(), g.columnIndices().deviceData(),
                                 g.values().deviceData(), map, workspace.prefix_.data(),
                                 workspace.inputKeys_.data(), workspace.inputValues_.data(),
                                 workspace.bits(rows));
    return compress(workspace, rows, nnz, Maximum());
}
/** @brief 按聚合编号写入最多四个成员槽。
 * @param n 本阶段工作维度。
 * @param map 细到粗聚合映射，规范化时允许为空。
 * @param sizes 各聚合成员数。
 * @param members 排序成员表。
 * @pre 当前设备正确，数组有效，维度与本阶段匹配，结构由上一阶段验证。
 * @return 无返回值。
 * @details 整数原子计数只决定临时槽位；随后排序保证最终成员次序确定。
 * @code slotMembers(n,map,sizes,members); @endcode */
__global__ void slotMembers(int n, const int *map, int *sizes, int *members) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int c = map[i], slot = atomicAdd(sizes + c, 1);
    members[size_t(c) * 4 + slot] = i;
}
/** @brief 对每个聚合的最多四个成员按细点编号排序。
 * @param n 本阶段工作维度。
 * @param sizes 各聚合成员数。
 * @param members 排序成员表。
 * @pre 当前设备正确，数组有效，维度与本阶段匹配，结构由上一阶段验证。
 * @return 无返回值。
 * @details 固定小容量使用插入排序，与原子填充的到达顺序无关。
 * @code sortMembers(n,sizes,members); @endcode */
__global__ void sortMembers(int n, const int *sizes, int *members) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n)
        return;
    int *m = members + size_t(c) * 4;
    for (int i = 1; i < sizes[c]; ++i) {
        int v = m[i], j = i;
        while (j > 0 && m[j - 1] > v) {
            m[j] = m[j - 1];
            --j;
        }
        m[j] = v;
    }
}
/** @brief 重建聚合大小及排序成员槽。
 * @param n 本阶段工作维度。
 * @param coarse 粗层维度或粗向量。
 * @param map 细到粗聚合映射，规范化时允许为空。
 * @param workspace 复用 GPU 工作区。
 * @pre 当前设备正确，数组有效，维度与本阶段匹配，结构由上一阶段验证。
 * @return 无返回值。
 * @details 每轮吸收后重建，使下一轮始终读取完整的新快照。
 * @code members(n,coarse,map,workspace); @endcode */
void members(int n, int coarse, const int *map, Workspace &workspace) {
    workspace.sizes_.reserve(size_t(coarse) + 1);
    workspace.memberSlots_.reserve(size_t(coarse) * 4);
    checkCuda(cudaMemset(workspace.sizes_.data(), 0, (size_t(coarse) + 1) * sizeof(int)), "AMG member sizes");
    slotMembers<<<grid(n), block>>>(n, map, workspace.sizes_.data(), workspace.memberSlots_.data());
    sortMembers<<<grid(coarse), block>>>(coarse, workspace.sizes_.data(), workspace.memberSlots_.data());
}
/** @brief 单点基于轮前快照向大小为二或三的聚合提议，强度相同取较小目标编号。
 * @param n 本阶段维度或条目数。
 * @param rowOffsets 当前设备上的rowOffsets数组或参数。
 * @param columnIndices 当前设备上的columnIndices数组或参数。
 * @param weights 当前设备上的weights数组或参数。
 * @param map 当前设备上的map数组或参数。
 * @param sizes 当前设备上的sizes数组或参数。
 * @param proposal 当前设备上的proposal数组或参数。
 * @param scores 当前设备上的scores数组或参数。
 * @pre 上一阶段的结构与维度有效，当前设备正确，各数组容量满足对应阶段要求。
 * @return 无返回值。
 * @code proposeAbsorption<<<blocks,256>>>(n,rowOffsets,columnIndices,weights,map,sizes,proposal,scores);
 * @endcode */
__global__ void proposeAbsorption(int n, const int *rowOffsets, const int *columnIndices,
                                  const double *weights, const int *map, const int *sizes, int *proposal,
                                  double *scores) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int best = -1;
    double score = 0;
    if (sizes[map[i]] == 1)
        for (int j = rowOffsets[i]; j < rowOffsets[i + 1]; ++j) {
            int c = map[columnIndices[j]];
            double s = weights[j];
            if (sizes[c] < 2 || sizes[c] >= 4 || s <= 0)
                continue;
            if (s > score || (s == score && (best < 0 || c < best))) {
                best = c;
                score = s;
            }
        }
    proposal[i] = best;
    scores[i] = score;
}
/** @brief 目标按剩余容量接收单点来源，按强度降序及细点编号升序排序，并写入新映射。
 * @param coarse 本阶段维度或条目数。
 * @param rowOffsets 当前设备上的rowOffsets数组或参数。
 * @param columnIndices 当前设备上的columnIndices数组或参数。
 * @param sizes 当前设备上的sizes数组或参数。
 * @param members 当前设备上的members数组或参数。
 * @param proposal 当前设备上的proposal数组或参数。
 * @param scores 当前设备上的scores数组或参数。
 * @param nextMap 当前设备上的nextMap数组或参数。
 * @pre 上一阶段的结构与维度有效，当前设备正确，各数组容量满足对应阶段要求。
 * @return 无返回值。
 * @details 通过去重避免同一来源被多个成员的邻接重复接收，最多接收两点，禁止轮内级联。
 * @code
 * acceptAbsorption<<<blocks,256>>>(coarse,rowOffsets,columnIndices,sizes,members,proposal,scores,nextMap);
 * @endcode */
__global__ void acceptAbsorption(int coarse, const int *rowOffsets, const int *columnIndices,
                                 const int *sizes, const int *members, const int *proposal,
                                 const double *scores, int *nextMap) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= coarse || sizes[c] < 2 || sizes[c] >= 4)
        return;
    int selected[2] = {-1, -1};
    double weights[2] = {-1, -1};
    const int capacity = 4 - sizes[c];
    for (int k = 0; k < sizes[c]; ++k) {
        int row = members[size_t(c) * 4 + k];
        for (int j = rowOffsets[row]; j < rowOffsets[row + 1]; ++j) {
            int i = columnIndices[j];
            if (proposal[i] != c || i == selected[0] || i == selected[1])
                continue;
            double s = scores[i];
            for (int q = 0; q < capacity; ++q)
                if (s > weights[q] || (s == weights[q] && i < selected[q])) {
                    for (int r = capacity - 1; r > q; --r) {
                        selected[r] = selected[r - 1];
                        weights[r] = weights[r - 1];
                    }
                    selected[q] = i;
                    weights[q] = s;
                    break;
                }
        }
    }
    for (int q = 0; q < capacity; ++q)
        if (selected[q] >= 0)
            nextMap[selected[q]] = c;
}
/** @brief 标记非空聚合，供前缀扫描压缩空编号。
 * @param n 本阶段工作维度。
 * @param sizes 各聚合成员数。
 * @param flags 扫描或校验标志。
 * @pre 当前设备正确，数组有效，维度与本阶段匹配，结构由上一阶段验证。
 * @return 无返回值。
 * @details 包含额外零哨兵，不直接修改当前细到粗映射。
 * @code occupied(n,sizes,flags); @endcode */
__global__ void occupied(int n, const int *sizes, int *flags) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c <= n)
        flags[c] = (c < n && sizes[c] > 0) ? 1 : 0;
}
/** @brief 把固定四槽成员表按扫描偏移压缩为有序成员 CSR，并累计规模直方图。
 * @param coarse 本阶段维度或条目数。
 * @param sizes 当前设备上的sizes数组或参数。
 * @param slots 当前设备上的slots数组或参数。
 * @param offsets 当前设备上的offsets数组或参数。
 * @param out 当前设备上的out数组或参数。
 * @param histogram 当前设备上的histogram数组或参数。
 * @pre 上一阶段的结构与维度有效，当前设备正确，各数组容量满足对应阶段要求。
 * @return 无返回值。
 * @code packMembers<<<blocks,256>>>(coarse,sizes,slots,offsets,out,histogram); @endcode */
__global__ void packMembers(int coarse, const int *sizes, const int *slots, const int *offsets, int *out,
                            int *histogram) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= coarse)
        return;
    for (int k = 0; k < sizes[c]; ++k)
        out[offsets[c] + k] = slots[size_t(c) * 4 + k];
    if (histogram && sizes[c])
        atomicAdd(histogram + sizes[c] - 1, 1);
}
/** @brief 两阶段匹配并组合映射，再按快照执行两轮受容量约束的单点吸收。
 * @param graph 强图。@param matchingRounds 每阶段匹配次数。@param absorbSingletons 吸收开关。
 * @param workspace 工作区。@param histogram 输出四种聚合规模的计数，调用方先清零。
 * @pre 对称正权无自环图，当前设备正确。@return 完整 GPU 传输表。
 * @details 每轮吸收先按旧映射生成提议，目标在排序成员邻接中选择最多剩余容量个
 * 来源；接受写入 nextMap_，然后交换并重建成员表。没有轮内级联吸收。
 * 最后压缩空编号，生成成员 CSR，使限制按确定顺序求和且无需浮点原子加法。
 * @code auto t=aggregateImpl(graph,8,true,workspace,histogram); @endcode */
AggregateTransfer aggregateImpl(const Matrix &graph, int matchingRounds, bool absorbSingletons,
                                Workspace &workspace, int *histogram) {
    int n = int(graph.rowCount());
    AggregateTransfer t;
    t.fineRows_ = n;
    t.aggregate_ = CudaBuffer<int>(n);
    int first = pair(graph, matchingRounds, t.aggregate_.data(), workspace);

    auto contracted = contractGraph(graph, t.aggregate_.data(), first, workspace);
    WorkspaceBuffer<int> second(first);
    t.coarseRows_ = pair(contracted, matchingRounds, second.data(), workspace);
    compose<<<grid(n), block>>>(n, t.aggregate_.data(), second.data());
    members(n, t.coarseRows_, t.aggregate_.data(), workspace);
    if (absorbSingletons) {
        workspace.proposal_.reserve(n);
        workspace.proposalWeight_.reserve(n);
        if (workspace.nextMap_.buffer_.size() != size_t(n))
            workspace.nextMap_.buffer_ = CudaBuffer<int>(n);
        for (int round = 0; round < 2; ++round) {
            proposeAbsorption<<<grid(n), block>>>(
                n, graph.rowOffsets().deviceData(), graph.columnIndices().deviceData(),
                graph.values().deviceData(), t.aggregate_.data(), workspace.sizes_.data(),
                workspace.proposal_.data(), workspace.proposalWeight_.data());
            checkCuda(cudaMemcpy(workspace.nextMap_.data(), t.aggregate_.data(), size_t(n) * sizeof(int),
                                 cudaMemcpyDeviceToDevice),
                      "AMG absorption snapshot");
            acceptAbsorption<<<grid(t.coarseRows_), block>>>(
                t.coarseRows_, graph.rowOffsets().deviceData(), graph.columnIndices().deviceData(),
                workspace.sizes_.data(), workspace.memberSlots_.data(), workspace.proposal_.data(),
                workspace.proposalWeight_.data(), workspace.nextMap_.data());
            t.aggregate_.swap(workspace.nextMap_.buffer_);
            members(n, t.coarseRows_, t.aggregate_.data(), workspace);
        }
        int coarse = t.coarseRows_;
        workspace.flags_.reserve(size_t(coarse) + 1);
        workspace.prefix_.reserve(size_t(coarse) + 1);
        occupied<<<grid(coarse + 1), block>>>(coarse, workspace.sizes_.data(), workspace.flags_.data());
        workspace.scan(workspace.flags_.data(), workspace.prefix_.data(), coarse + 1);
        compose<<<grid(n), block>>>(n, t.aggregate_.data(), workspace.prefix_.data());
        checkCuda(cudaMemcpy(&t.coarseRows_, workspace.prefix_.data() + coarse, sizeof(int),
                             cudaMemcpyDeviceToHost),
                  "AMG absorbed count");
        members(n, t.coarseRows_, t.aggregate_.data(), workspace);
    }
    t.members_ = CudaBuffer<int>(n);
    t.offsets_ = CudaBuffer<int>(size_t(t.coarseRows_) + 1);
    workspace.scan(workspace.sizes_.data(), t.offsets_.data(), t.coarseRows_ + 1);
    packMembers<<<grid(t.coarseRows_), block>>>(t.coarseRows_, workspace.sizes_.data(),
                                                workspace.memberSlots_.data(), t.offsets_.data(),
                                                t.members_.data(), histogram);
    return t;
}

/** @brief 在单线程块共享内存中执行最多 64 阶的部分主元 LU。
 * @param n 阶数。@param rowOffsets CSR 偏移。@param columnIndices 列号。@param a 系数。
 * @param lu 输出因子。@param pivots 逐步主元。@param bad 奇异或非有限标志。
 * @pre n 在 1..64，规范 CSR，以 64 线程启动单个块。@return 无返回值。
 * @details 以矩阵最大绝对值缩放 64*epsilon 阈值。选主元、换行、消元间均块同步。
 * 失败状态由全部线程统一读取并退出，避免同步分歧。
 * @code factor<<<1,64>>>(n,rp,ci,a,lu,pivots,bad); @endcode */
__global__ void factor(int n, const int *rowOffsets, const int *columnIndices, const double *a, double *lu,
                       int *pivots, int *bad) {
    __shared__ double m[64 * 64], rowScale[64], scale;
    __shared__ int pivot, failed;
    int row = threadIdx.x;
    double maximum = 0;
    if (row < n) {
        for (int j = 0; j < n; ++j)
            m[row * n + j] = 0;
        for (int j = rowOffsets[row]; j < rowOffsets[row + 1]; ++j) {
            m[row * n + columnIndices[j]] = a[j];
            maximum = fmax(maximum, fabs(a[j]));
        }
    }
    rowScale[row] = maximum;
    __syncthreads();
    if (row == 0) {
        scale = 0;
        for (int i = 0; i < n; ++i)
            scale = fmax(scale, rowScale[i]);
        failed = 0;
        *bad = 0;
    }
    __syncthreads();
    for (int k = 0; k < n; ++k) {
        if (row == 0) {
            pivot = k;
            for (int i = k + 1; i < n; ++i)
                if (fabs(m[i * n + k]) > fabs(m[pivot * n + k]))
                    pivot = i;
            if (!isfinite(m[pivot * n + k]) || fabs(m[pivot * n + k]) <= 64 * 2.2204460492503131e-16 * scale)
                failed = 1;
            pivots[k] = pivot;
        }
        __syncthreads();
        if (failed) {
            if (row == 0)
                *bad = 1;
            return;
        }
        if (pivot != k && row < n) {
            double v = m[pivot * n + row];
            m[pivot * n + row] = m[k * n + row];
            m[k * n + row] = v;
        }
        __syncthreads();
        if (row > k && row < n) {
            m[row * n + k] /= m[k * n + k];
            for (int j = k + 1; j < n; ++j)
                m[row * n + j] -= m[row * n + k] * m[k * n + j];
        }
        __syncthreads();
    }
    for (int i = row; i < n * n; i += 64) {
        lu[i] = m[i];
        if (!isfinite(m[i]))
            atomicExch(bad, 1);
    }
}
/** @brief 使用缓存主元和 LU 在 GPU 上求解粗方程。
 * @param n 本阶段工作维度。
 * @param lu 缓存 LU。
 * @param pivots 逐步交换主元。
 * @param b 右端项。
 * @param x 输出近似解。
 * @pre 当前设备正确，数组有效，维度与本阶段匹配，结构由上一阶段验证。
 * @return 无返回值。
 * @details 单块先置换右端，再依次前后代入，共享 double 向量允许 rhs/x 别名。
 * @code triangular(n,lu,pivots,b,x); @endcode */
template <linalg::Scalar T>
__global__ void triangular(int n, const double *lu, const int *pivots, const T *b, T *x) {
    __shared__ double v[64];
    int row = threadIdx.x;
    if (row < n)
        v[row] = b[row];
    __syncthreads();
    if (row == 0)
        for (int k = 0; k < n; ++k) {
            double t = v[k];
            v[k] = v[pivots[k]];
            v[pivots[k]] = t;
        }
    __syncthreads();
    for (int k = 0; k < n; ++k) {
        if (row > k && row < n)
            v[row] -= lu[row * n + k] * v[k];
        __syncthreads();
    }
    for (int k = n - 1; k >= 0; --k) {
        if (row == k)
            v[row] /= lu[row * n + row];
        __syncthreads();
        if (row < k)
            v[row] -= lu[row * n + k] * v[k];
        __syncthreads();
    }
    if (row < n)
        x[row] = v[row];
}
/** @brief 按有序成员表逐聚合求和限制细向量。
 * @param n 本阶段工作维度。
 * @param rowOffsets CSR 行偏移。
 * @param members 排序成员表。
 * @param fine 细向量。
 * @param coarse 粗层维度或粗向量。
 * @pre 当前设备正确，数组有效，维度与本阶段匹配，结构由上一阶段验证。
 * @return 无返回值。
 * @details 每个粗点由单线程写入，不用浮点原子操作；运算次序由成员编号确定。
 * @code restrictKernel(n,rowOffsets,members,fine,coarse); @endcode */
template <linalg::Scalar T>
__global__ void restrictKernel(int n, const int *rowOffsets, const int *members, const T *fine, T *coarse) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T sum = 0;
        for (int j = rowOffsets[i]; j < rowOffsets[i + 1]; ++j)
            sum += fine[members[j]];
        coarse[i] = sum;
    }
}
/** @brief 按细到粗映射延拓并累加粗层修正。
 * @param n 本阶段工作维度。
 * @param map 细到粗聚合映射，规范化时允许为空。
 * @param coarse 粗层维度或粗向量。
 * @param fine 细向量。
 * @pre 当前设备正确，数组有效，维度与本阶段匹配，结构由上一阶段验证。
 * @return 无返回值。
 * @details 每个细点只读一个粗值并更新自身，保留细层当前近似解。
 * @code prolongKernel(n,map,coarse,fine); @endcode */
template <linalg::Scalar T> __global__ void prolongKernel(int n, const int *map, const T *coarse, T *fine) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        fine[i] += coarse[map[i]];
}

} // namespace

/** @brief 独占工作区，不含 AMG 的层次和控制逻辑。
 * @code GpuAggregation::Impl impl(false); @endcode */
struct GpuAggregation::Impl {
    Workspace workspace_; ///< 可复用 CUB 及匹配缓冲区。
    int device_;          ///< 创建时的设备。
    explicit Impl(bool compact) : workspace_(compact) {
        checkCuda(cudaGetDevice(&device_), "AMG workspace device");
    }
};
GpuAggregation::GpuAggregation(bool compactKeys) : impl_(std::make_unique<Impl>(compactKeys)) {}
GpuAggregation::~GpuAggregation() = default;
CsrMatrix<double> GpuAggregation::canonical(const CsrMatrix<double> &a) {
    linalg::detail::OperationScope scope(impl_->device_);
    if (!a.rowCount() || a.rowCount() >= size_t(INT_MAX) || a.rowCount() != a.columnCount() ||
        a.nonZeroCount() > size_t(INT_MAX / 2) || a.rowOffsets().deviceId() != impl_->device_)
        throw std::invalid_argument("AMG requires nonempty square int32-capacity CSR on workspace device");
    auto &workspace = impl_->workspace_;
    checkCuda(cudaMemset(workspace.count_.data(), 0, sizeof(int)), "CSR canonical flag");
    checkCanonical<<<grid(int(a.rowCount())), block>>>(
        int(a.rowCount()), int(a.nonZeroCount()), a.rowOffsets().deviceData(), a.columnIndices().deviceData(),
        a.values().deviceData(), workspace.count_.data());
    const int state = workspace.scalar();
    if (state & 2)
        throw std::invalid_argument("AMG invalid CSR or nonfinite coefficient");
    return state == 0 ? a : multigrid::galerkin(a, nullptr, int(a.rowCount()), workspace);
}
void GpuAggregation::validate(const CsrMatrix<double> &a) {
    linalg::detail::OperationScope scope(impl_->device_);
    validateMatrix(a, impl_->workspace_);
}
CsrMatrix<double> GpuAggregation::strength(const CsrMatrix<double> &a, double threshold) {
    if (!std::isfinite(threshold) || threshold < 0 || threshold > 1)
        throw std::invalid_argument("Invalid strength threshold");
    linalg::detail::OperationScope scope(impl_->device_);
    validateMatrix(a, impl_->workspace_);
    return strengthGraph(a, threshold, impl_->workspace_);
}
AggregateTransfer GpuAggregation::aggregate(const CsrMatrix<double> &graph, int rounds,
                                            bool absorbSingletons) {
    if (rounds < 1 || rounds > 8 || !graph.rowCount())
        throw std::invalid_argument("Invalid matching configuration");
    linalg::detail::OperationScope scope(impl_->device_);
    CudaBuffer<int> histogram(4);
    checkCuda(cudaMemset(histogram.data(), 0, 4 * sizeof(int)), "AMG histogram");
    auto transfer = aggregateImpl(graph, rounds, absorbSingletons, impl_->workspace_, histogram.data());
    histogram.copyToHost(transfer.histogram_);
    checkCuda(cudaGetLastError(), "AMG aggregation");
    return transfer;
}
CsrMatrix<double> GpuAggregation::galerkin(const CsrMatrix<double> &a, const AggregateTransfer &t) {
    if (a.rowCount() != size_t(t.fineRows_) || t.coarseRows_ < 1)
        throw std::invalid_argument("AMG transfer dimension mismatch");
    linalg::detail::OperationScope scope(impl_->device_);
    return multigrid::galerkin(a, t.aggregate_.data(), t.coarseRows_, impl_->workspace_);
}
/** @brief 验证传输向量的设备、维度、策略及非别名约束。
 * @param t 传输。@param f 细向量。@param c 粗向量。
 * @pre 映射由本模块生成。@return 无返回值。@code validateTransfer(t,f,c); @endcode */
template <linalg::Scalar T>
void validateTransfer(const AggregateTransfer &t, const Vector<T> &f, const Vector<T> &c) {
    if (f.size() != size_t(t.fineRows_) || c.size() != size_t(t.coarseRows_) ||
        f.deviceId() != t.aggregate_.deviceId() || c.deviceId() != t.aggregate_.deviceId() ||
        f.memoryType() != MemoryType::Global || c.memoryType() != MemoryType::Global || f.data() == c.data())
        throw std::invalid_argument("Invalid aggregate transfer vectors");
}
template <linalg::Scalar T>
void restrictAggregates(const AggregateTransfer &t, const Vector<T> &fine, Vector<T> &coarse) {
    validateTransfer(t, fine, coarse);
    linalg::detail::OperationScope scope(t.aggregate_.deviceId());
    restrictKernel<<<grid(t.coarseRows_), block>>>(t.coarseRows_, t.offsets_.data(), t.members_.data(),
                                                   fine.deviceData(), coarse.deviceData());
    checkCuda(cudaGetLastError(), "AMG restriction");
    checkCuda(cudaDeviceSynchronize(), "AMG restriction completion");
}
template <linalg::Scalar T>
void prolongAggregates(const AggregateTransfer &t, const Vector<T> &coarse, Vector<T> &fine) {
    validateTransfer(t, fine, coarse);
    linalg::detail::OperationScope scope(t.aggregate_.deviceId());
    prolongKernel<<<grid(t.fineRows_), block>>>(t.fineRows_, t.aggregate_.data(), coarse.deviceData(),
                                                fine.deviceData());
    checkCuda(cudaGetLastError(), "AMG prolongation");
    checkCuda(cudaDeviceSynchronize(), "AMG prolongation completion");
}
CoarseFactor factorCoarse(const CsrMatrix<double> &a) {
    if (!a.rowCount() || a.rowCount() > 64 || a.rowCount() != a.columnCount())
        throw std::invalid_argument("GPU LU requires order 1..64");
    linalg::detail::OperationScope scope(a.rowOffsets().deviceId());
    CoarseFactor result;
    result.rows_ = int(a.rowCount());
    result.lu_ = CudaBuffer<double>(a.rowCount() * a.rowCount());
    result.pivots_ = CudaBuffer<int>(a.rowCount());
    CudaBuffer<int> bad(1);
    factor<<<1, 64>>>(result.rows_, a.rowOffsets().deviceData(), a.columnIndices().deviceData(),
                      a.values().deviceData(), result.lu_.data(), result.pivots_.data(), bad.data());
    int flag{};
    bad.copyToHost(std::span<int>(&flag, 1));
    result.singular_ = flag != 0;
    result.factored_ = !result.singular_;
    checkCuda(cudaGetLastError(), "AMG coarse factor");
    return result;
}
template <linalg::Scalar T> void solveCoarse(const CoarseFactor &factor, const Vector<T> &rhs, Vector<T> &x) {
    if (!factor.factored_ || rhs.size() != size_t(factor.rows_) || x.size() != rhs.size() ||
        rhs.deviceId() != factor.lu_.deviceId() || x.deviceId() != factor.lu_.deviceId() ||
        rhs.memoryType() != MemoryType::Global || x.memoryType() != MemoryType::Global)
        throw std::invalid_argument("Invalid GPU coarse solve");
    linalg::detail::OperationScope scope(factor.lu_.deviceId());
    triangular<<<1, 64>>>(factor.rows_, factor.lu_.data(), factor.pivots_.data(), rhs.deviceData(),
                          x.deviceData());
    checkCuda(cudaGetLastError(), "AMG coarse solve");
    checkCuda(cudaDeviceSynchronize(), "AMG coarse completion");
}
/** @brief 逐元素转换 GPU 数组。@param n 长度。@param in 来源。@param out 输出。
 * @pre 可表示转换，同设备。@return 无返回值。@code convert<<<1,256>>>(n,in,out); @endcode */
template <BufferElement A, BufferElement B>
    requires std::convertible_to<A, B>
__global__ void convert(size_t n, const A *in, B *out) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = static_cast<B>(in[i]);
}
/** @brief 转换全部 CSR 数组。@param a 输入。@pre 维度可表示。
 * @return 目标矩阵。@code convertMatrix<float,int>(a); @endcode */
template <linalg::Scalar T, CsrIndex I, linalg::Scalar U, CsrIndex J>
CsrMatrix<T, I> convertMatrix(const CsrMatrix<U, J> &a) {
    if (a.rowCount() >= size_t(INT_MAX) || a.nonZeroCount() > size_t(INT_MAX / 2))
        throw std::length_error("AMG native index capacity exceeded");
    CudaBuffer<I> rowOffsets(a.rowCount() + 1), columnIndices(a.nonZeroCount());
    CudaBuffer<T> entries(a.nonZeroCount());
    convert<<<grid(int(rowOffsets.size())), block>>>(rowOffsets.size(), a.rowOffsets().deviceData(),
                                                     rowOffsets.data());
    if (!columnIndices.empty()) {
        convert<<<grid(int(columnIndices.size())), block>>>(
            columnIndices.size(), a.columnIndices().deviceData(), columnIndices.data());
        convert<<<grid(int(entries.size())), block>>>(entries.size(), a.values().deviceData(),
                                                      entries.data());
    }
    checkCuda(cudaGetLastError(), "AMG matrix conversion");
    return CsrMatrix<T, I>::fromBuffers(a.rowCount(), a.columnCount(), std::move(rowOffsets),
                                        std::move(columnIndices), std::move(entries));
}
template <linalg::Scalar T, CsrIndex I> CsrMatrix<double> toNativeMatrix(const CsrMatrix<T, I> &a) {
    return convertMatrix<double, int>(a);
}
template <linalg::Scalar T, CsrIndex I> CsrMatrix<T, I> fromNativeMatrix(const CsrMatrix<double> &a) {
    return convertMatrix<T, I>(a);
}
#define THERMOFEM_TRANSFER(T)                                                                                \
    template void restrictAggregates<T>(const AggregateTransfer &, const Vector<T> &, Vector<T> &);          \
    template void prolongAggregates<T>(const AggregateTransfer &, const Vector<T> &, Vector<T> &);           \
    template void solveCoarse<T>(const CoarseFactor &, const Vector<T> &, Vector<T> &);
THERMOFEM_TRANSFER(float)
THERMOFEM_TRANSFER(double)
#undef THERMOFEM_TRANSFER
#define THERMOFEM_CONVERSION(T, I)                                                                           \
    template CsrMatrix<double> toNativeMatrix<T, I>(const CsrMatrix<T, I> &);                                \
    template CsrMatrix<T, I> fromNativeMatrix<T, I>(const CsrMatrix<double> &);
THERMOFEM_CONVERSION(float, int)
THERMOFEM_CONVERSION(double, int)
THERMOFEM_CONVERSION(float, std::int64_t)
THERMOFEM_CONVERSION(double, std::int64_t)
#undef THERMOFEM_CONVERSION
} // namespace thermofem::multigrid

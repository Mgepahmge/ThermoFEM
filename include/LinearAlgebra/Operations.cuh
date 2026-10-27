/** @file Operations.cuh
 * @brief 封装实数向量、行主序稠密矩阵与 CSR 矩阵的同步线性代数运算。
 */
#pragma once
#include <LinearAlgebra/Backend.cuh>
#include <DataStructure/DenseMatrix.cuh>
#include <DataStructure/Vector.cuh>
#include <algorithm>
#include <initializer_list>
#include <limits>
#include <utility>

/** @brief 封装由 cuBLAS 与 cuSPARSE 执行的基础线性代数运算。 */
namespace thermofem::linalg {
namespace detail {
/** @brief 检查操作数设备是否一致。@param devices 设备编号，忽略 -1。
 * @pre 无额外约束，不一致时抛异常。@return 公共设备或 -1。
 * @code auto device = commonDevice({x.deviceId(), y.deviceId()}); @endcode
 */
inline int commonDevice(std::initializer_list<int> devices) {
    int result = -1;
    for (const int device : devices) {
        if (device >= 0) {
            if (result >= 0 && result != device) { throw std::invalid_argument("Operands belong to different CUDA devices"); }
            result = device;
        }
    }
    return result;
}
/** @brief 检查 32 位库接口的尺寸。@param count 尺寸。
 * @pre 无额外约束，超范围抛 length_error。@return int 尺寸。
 * @code const int size = librarySize(vector.size()); @endcode
 */
inline int librarySize(std::size_t count) {
    if (!std::in_range<int>(count)) { throw std::length_error("Operation exceeds 32-bit library interface range"); }
    return static_cast<int>(count);
}
/** @brief 下载存储数据。@tparam T 元素类型。@param buffer 来源。@pre 外部操作已同步。
 * @return 主机数组。@code auto values = hostCopy(buffer); @endcode
 */
template<BufferElement T> std::vector<T> hostCopy(const CudaBuffer<T>& buffer) {
    std::vector<T> values(buffer.size());
    buffer.copyToHost(values);
    return values;
}
/** @brief 创建 global 深拷贝。@tparam T 元素类型。@param buffer 来源。
 * @pre 当前设备为所属设备。@return 设备缓冲区。@code auto copy = deviceCopy(buffer); @endcode
 */
template<BufferElement T> CudaBuffer<T> deviceCopy(const CudaBuffer<T>& buffer) {
    CudaBuffer<T> result(buffer.size());
    result.copyDataFrom(buffer);
    return result;
}
/** @brief 显式下载向量的内部存储。@tparam T 元素类型。@param vector 来源。
 * @pre 外部访问已同步。@return 主机数组。@code auto host = hostCopy(x); @endcode */
template<BufferElement T> std::vector<T> hostCopy(const Vector<T>& vector) { return hostCopy(vector.values()); }
/** @brief 显式复制向量的内部存储至 global。@tparam T 元素类型。@param vector 来源。
 * @pre 当前设备为所属设备。@return 设备缓冲区。@code auto copy = deviceCopy(x); @endcode */
template<BufferElement T> CudaBuffer<T> deviceCopy(const Vector<T>& vector) { return deviceCopy(vector.values()); }
/** @brief 将设备输出清零。@tparam T 精度。@param pointer 设备地址。@param count 元素数。
 * @pre 地址有效且属于当前设备。@return 无返回值。@code zero(pointer, 3); @endcode
 */
template<Scalar T> void zero(T* pointer, std::size_t count) {
    if (count == 0) { return; }
    auto status = cudaMemset(pointer, 0, thermofem::detail::checkedProduct(count, sizeof(T)));
    if (status != cudaSuccess) { throw CudaError(status, "cudaMemset"); }
    status = cudaStreamSynchronize(nullptr);
    if (status != cudaSuccess) { throw CudaError(status, "cudaStreamSynchronize"); }
}
/** @brief 规范化 CSR 并转换索引，主机排序及合并重复项后上传。
 * @tparam Out 结果索引。@tparam T 精度。@tparam I 来源索引。@param matrix 输入。
 * @pre 所有尺寸均能由 Out 表示，当前设备正确。@return 有序且无重复的 global CSR。
 * @code auto input = canonical<int>(matrix); @endcode
 */
template<CsrIndex Out, Scalar T, CsrIndex I>
CsrMatrix<T, Out> canonical(const CsrMatrix<T, I>& matrix) {
    if (!std::in_range<Out>(matrix.rowCount()) || !std::in_range<Out>(matrix.columnCount())
        || !std::in_range<Out>(matrix.nonZeroCount())) {
        throw std::length_error("Sparse operation exceeds its structural index range");
    }
    const auto offsets = hostCopy(matrix.rowOffsets());
    const auto columns = hostCopy(matrix.columnIndices());
    const auto values = hostCopy(matrix.values());
    std::vector<Out> outOffsets(matrix.rowCount() + 1, 0), outColumns;
    std::vector<T> outValues;
    for (std::size_t row = 0; row < matrix.rowCount(); ++row) {
        std::vector<std::pair<Out, T>> entries;
        for (auto index = offsets[row]; index < offsets[row + 1]; ++index) {
            const auto position = static_cast<std::size_t>(index);
            entries.emplace_back(static_cast<Out>(columns[position]), values[position]);
        }
        std::stable_sort(entries.begin(), entries.end(), [](const std::pair<Out, T>& left, const std::pair<Out, T>& right) {
            return left.first < right.first;
        });
        for (const auto& [column, value] : entries) {
            if (outColumns.size() > static_cast<std::size_t>(outOffsets[row]) && outColumns.back() == column) {
                outValues.back() += value;
            } else { outColumns.push_back(column); outValues.push_back(value); }
        }
        outOffsets[row + 1] = static_cast<Out>(outValues.size());
    }
    return CsrMatrix<T, Out>(matrix.rowCount(), matrix.columnCount(), outOffsets, outColumns, outValues);
}
/** @brief 构造空稀疏结果，保留形状与策略。@tparam T 精度。@tparam I 索引类型。
 * @param rows 行数。@param columns 列数。@param strategy 策略。@pre 尺寸可表示。
 * @return 空 CSR。@code auto result = emptySparse<double, int>(2, 3, GlobalMemory{}); @endcode
 */
template<Scalar T, CsrIndex I>
CsrMatrix<T, I> emptySparse(std::size_t rows, std::size_t columns, MemoryStrategy strategy) {
    if (rows == 0 && columns == 0) {
        CsrMatrix<T, I> result;
        result.changeMemoryType(strategy);
        return result;
    }
    if (!std::in_range<I>(rows) || !std::in_range<I>(columns)) { throw std::length_error("CSR size exceeds index range"); }
    const std::vector<I> offsets(rows + 1, 0);
    return CsrMatrix<T, I>(rows, columns, offsets, {}, {}, strategy);
}
/** @brief 将库的主机 CSR 结果恢复为目标索引与策略。
 * @tparam T 精度。@tparam I 目标索引。@param rows 行数。@param columns 列数。
 * @param data 主机结果。@param strategy 策略。@pre 尺寸合法。@return CSR 结果。
 * @code auto result = restoreSparse<double, long>(2, 3, data, GlobalMemory{}); @endcode
 */
template<Scalar T, CsrIndex I>
CsrMatrix<T, I> restoreSparse(std::size_t rows, std::size_t columns, const SparseHostData<T>& data, MemoryStrategy strategy) {
    const std::vector<I> offsets(data.offsets_.begin(), data.offsets_.end());
    const std::vector<I> indices(data.columns_.begin(), data.columns_.end());
    return CsrMatrix<T, I>(rows, columns, offsets, indices, data.values_, strategy);
}
} // 命名空间 detail

/** @brief 计算向量缩放。@tparam T float 或 double。@param vector 输入。@param factor 系数。
 * @pre 外部异步访问已结束，元素数不超过 INT_MAX。@return 沿用输入策略的新向量。
 * @code auto result = thermofem::linalg::scale(vector, 2.0); @endcode
 */
template<Scalar T> Vector<T> scale(const Vector<T>& vector, std::type_identity_t<T> factor) {
    const int count = detail::librarySize(vector.size());
    detail::OperationScope scope(vector.deviceId());
    Vector<T> result(vector.size());
    result.copyDataFrom(vector);
    if (count) { detail::blasScale(count, factor, result.deviceData()); }
    result.changeMemoryType(vector.memoryStrategy());
    return result;
}
/** @brief 计算 x + beta*y。@tparam T 浮点精度。@param x 左向量。@param y 右向量。@param beta 右系数。
 * @pre 长度和设备一致，外部访问已同步，尺寸不超过 INT_MAX。@return 沿用 x 策略的新向量。
 * @code auto result = thermofem::linalg::add(x, y); @endcode
 */
template<Scalar T> Vector<T> add(const Vector<T>& x, const Vector<T>& y, std::type_identity_t<T> beta = T{1}) {
    if (x.size() != y.size()) { throw std::invalid_argument("Vector sizes must match"); }
    const int count = detail::librarySize(x.size());
    detail::OperationScope scope(detail::commonDevice({x.deviceId(), y.deviceId()}));
    Vector<T> result(x.size());
    result.copyDataFrom(x);
    if (count) { auto input = detail::deviceCopy<T>(y); detail::blasAxpy(count, beta, input.deviceData(), result.deviceData()); }
    result.changeMemoryType(x.memoryStrategy());
    return result;
}
/** @brief 计算向量差。@tparam T 精度。@param x 左向量。@param y 右向量。
 * @pre 同 add()。@return 沿用 x 策略的新向量。@code auto result = thermofem::linalg::subtract(x, y); @endcode
 */
template<Scalar T> Vector<T> subtract(const Vector<T>& x, const Vector<T>& y) { return add(x, y, T{-1}); }
/** @brief 计算 alpha*x+y。@tparam T 精度。@param alpha 系数。@param x 输入。@param y 输入。
 * @pre 同 add()。@return 沿用 y 策略的新向量，输入不改变。
 * @code auto result = thermofem::linalg::axpy(2.0, x, y); @endcode
 */
template<Scalar T> Vector<T> axpy(std::type_identity_t<T> alpha, const Vector<T>& x, const Vector<T>& y) { return add(y, x, alpha); }
/** @brief 计算实数内积。@tparam T 精度。@param x 左向量。@param y 右向量。
 * @pre 长度和设备一致，外部访问已同步，尺寸不超过 INT_MAX。@return 内积，空向量为 0。
 * @code auto value = thermofem::linalg::dot(x, y); @endcode
 */
template<Scalar T> T dot(const Vector<T>& x, const Vector<T>& y) {
    if (x.size() != y.size()) { throw std::invalid_argument("Vector sizes must match"); }
    const int count = detail::librarySize(x.size());
    detail::OperationScope scope(detail::commonDevice({x.deviceId(), y.deviceId()}));
    if (!count) { return T{}; }
    auto left = detail::deviceCopy<T>(x), right = detail::deviceCopy<T>(y);
    return detail::blasDot(count, left.deviceData(), right.deviceData());
}
/** @brief 以向量乘法表示实数内积。@tparam T 精度。@param x 左向量。@param y 右向量。
 * @pre 同 dot()。@return 内积标量。@code auto value = thermofem::linalg::multiply(x, y); @endcode
 */
template<Scalar T> T multiply(const Vector<T>& x, const Vector<T>& y) { return dot(x, y); }
/** @brief 计算向量二范数。@tparam T 精度。@param vector 输入。
 * @pre 外部访问已同步，尺寸不超过 INT_MAX。@return 二范数，空向量为 0。
 * @code auto value = thermofem::linalg::norm2(vector); @endcode
 */
template<Scalar T> T norm2(const Vector<T>& vector) {
    const int count = detail::librarySize(vector.size());
    if (!count) { return T{}; }
    detail::OperationScope scope(vector.deviceId());
    auto input = detail::deviceCopy<T>(vector);
    return detail::blasNorm(count, input.deviceData(), false);
}
/** @brief 计算向量一范数（绝对值之和）。@tparam T 精度。@param vector 输入。
 * @pre 同 norm2()。@return 一范数，空向量为 0。@code auto value = thermofem::linalg::norm1(vector); @endcode
 */
template<Scalar T> T norm1(const Vector<T>& vector) {
    const int count = detail::librarySize(vector.size());
    if (!count) { return T{}; }
    detail::OperationScope scope(vector.deviceId());
    auto input = detail::deviceCopy<T>(vector);
    return detail::blasNorm(count, input.deviceData(), true);
}

/** @brief 缩放稠密矩阵。@tparam T 精度。@param matrix 输入。@param factor 系数。
 * @pre 元素数不超过 INT_MAX，外部访问已同步。@return 沿用输入策略的新矩阵。
 * @code auto result = thermofem::linalg::scale(matrix, 2.0); @endcode
 */
template<Scalar T> DenseMatrix<T> scale(const DenseMatrix<T>& matrix, std::type_identity_t<T> factor) {
    const int count = detail::librarySize(matrix.values().size());
    detail::OperationScope scope(matrix.values().deviceId());
    DenseMatrix<T> result(matrix.rowCount(), matrix.columnCount());
    result.copyValuesFrom(matrix.values());
    if (count) { detail::blasScale(count, factor, result.deviceData()); }
    result.changeMemoryType(matrix.values().memoryStrategy());
    return result;
}
/** @brief 计算稠密矩阵 a+beta*b。@tparam T 精度。@param a 左矩阵。@param b 右矩阵。@param beta 右系数。
 * @pre 形状和设备一致，元素数不超过 INT_MAX，外部访问已同步。@return 沿用 a 策略的新矩阵。
 * @code auto result = thermofem::linalg::add(a, b); @endcode
 */
template<Scalar T> DenseMatrix<T> add(const DenseMatrix<T>& a, const DenseMatrix<T>& b, std::type_identity_t<T> beta = T{1}) {
    if (a.rowCount() != b.rowCount() || a.columnCount() != b.columnCount()) { throw std::invalid_argument("Dense matrix shapes must match"); }
    const int count = detail::librarySize(a.values().size());
    detail::OperationScope scope(detail::commonDevice({a.values().deviceId(), b.values().deviceId()}));
    DenseMatrix<T> result(a.rowCount(), a.columnCount());
    result.copyValuesFrom(a.values());
    if (count) { auto input = detail::deviceCopy(b.values()); detail::blasAxpy(count, beta, input.deviceData(), result.deviceData()); }
    result.changeMemoryType(a.values().memoryStrategy());
    return result;
}
/** @brief 计算稠密矩阵差。@tparam T 精度。@param a 左矩阵。@param b 右矩阵。
 * @pre 同 add()。@return 沿用 a 策略的新矩阵。@code auto result = thermofem::linalg::subtract(a, b); @endcode
 */
template<Scalar T> DenseMatrix<T> subtract(const DenseMatrix<T>& a, const DenseMatrix<T>& b) { return add(a, b, T{-1}); }
/** @brief 转置稠密矩阵。@tparam T 精度。@param matrix 输入。
 * @pre 行列数不超过 INT_MAX，外部访问已同步。@return 沿用输入策略的行主序转置矩阵。
 * @code auto result = thermofem::linalg::transpose(matrix); @endcode
 */
template<Scalar T> DenseMatrix<T> transpose(const DenseMatrix<T>& matrix) {
    const int rows = detail::librarySize(matrix.rowCount()), columns = detail::librarySize(matrix.columnCount());
    detail::OperationScope scope(matrix.values().deviceId());
    DenseMatrix<T> result(matrix.columnCount(), matrix.rowCount());
    if (rows && columns) { auto input = detail::deviceCopy(matrix.values()); detail::blasTranspose(rows, columns, input.deviceData(), result.deviceData()); }
    result.changeMemoryType(matrix.values().memoryStrategy());
    return result;
}
/** @brief 计算稠密矩阵向量积。@tparam T 精度。@param matrix 左矩阵。@param vector 右向量。
 * @pre 内维度和设备一致，维度不超过 INT_MAX，外部访问已同步。@return 沿用矩阵策略的新向量。
 * @code auto result = thermofem::linalg::multiply(matrix, vector); @endcode
 */
template<Scalar T> Vector<T> multiply(const DenseMatrix<T>& matrix, const Vector<T>& vector) {
    if (matrix.columnCount() != vector.size()) { throw std::invalid_argument("Matrix-vector inner dimensions must match"); }
    const int rows = detail::librarySize(matrix.rowCount()), columns = detail::librarySize(matrix.columnCount());
    detail::OperationScope scope(detail::commonDevice({matrix.values().deviceId(), vector.deviceId()}));
    Vector<T> result(matrix.rowCount());
    if (rows && columns) {
        auto a = detail::deviceCopy(matrix.values()), x = detail::deviceCopy<T>(vector);
        detail::blasGemv(rows, columns, a.deviceData(), x.deviceData(), result.deviceData());
    } else { detail::zero(result.deviceData(), result.size()); }
    result.changeMemoryType(matrix.values().memoryStrategy());
    return result;
}
/** @brief 计算稠密矩阵乘积。@tparam T 精度。@param a 左矩阵。@param b 右矩阵。
 * @pre 内维度和设备一致，维度不超过 INT_MAX，外部访问已同步。@return 沿用 a 策略的新矩阵。
 * @code auto result = thermofem::linalg::multiply(a, b); @endcode
 */
template<Scalar T> DenseMatrix<T> multiply(const DenseMatrix<T>& a, const DenseMatrix<T>& b) {
    if (a.columnCount() != b.rowCount()) { throw std::invalid_argument("Matrix inner dimensions must match"); }
    const int rows = detail::librarySize(a.rowCount()), columns = detail::librarySize(b.columnCount()), inner = detail::librarySize(a.columnCount());
    detail::OperationScope scope(detail::commonDevice({a.values().deviceId(), b.values().deviceId()}));
    DenseMatrix<T> result(a.rowCount(), b.columnCount());
    if (rows && columns && inner) {
        auto left = detail::deviceCopy(a.values()), right = detail::deviceCopy(b.values());
        detail::blasGemm(rows, columns, inner, left.deviceData(), right.deviceData(), result.deviceData());
    } else { detail::zero(result.deviceData(), result.values().size()); }
    result.changeMemoryType(a.values().memoryStrategy());
    return result;
}
/** @brief 计算稠密矩阵 Frobenius 范数。@tparam T 精度。@param matrix 输入。
 * @pre 元素数不超过 INT_MAX，外部访问已同步。@return 范数。
 * @code auto value = thermofem::linalg::frobeniusNorm(matrix); @endcode
 */
template<Scalar T> T frobeniusNorm(const DenseMatrix<T>& matrix) {
    const int count = detail::librarySize(matrix.values().size());
    if (!count) { return T{}; }
    detail::OperationScope scope(matrix.values().deviceId());
    auto input = detail::deviceCopy(matrix.values());
    return detail::blasNorm(count, input.deviceData(), false);
}

/** @brief 缩放 CSR 数值，保持原稀疏结构（含重复项）不变。@tparam T 精度。@tparam I 索引。
 * @param matrix 输入。@param factor 系数。@pre 非零数不超过 INT_MAX，外部访问已同步。
 * @return 沿用输入策略的新 CSR；数值缩放由 cuBLAS 完成。
 * @code auto result = thermofem::linalg::scale(matrix, 2.0); @endcode
 */
template<Scalar T, CsrIndex I> CsrMatrix<T, I> scale(const CsrMatrix<T, I>& matrix, std::type_identity_t<T> factor) {
    const int count = detail::librarySize(matrix.nonZeroCount());
    detail::OperationScope scope(matrix.rowOffsets().deviceId());
    auto result = matrix;
    if (count) {
        auto input = detail::deviceCopy(matrix.values());
        detail::blasScale(count, factor, input.deviceData());
        result.copyValuesFromHost(detail::hostCopy(input));
    }
    return result;
}
/** @brief 计算 CSR 与向量乘积（SpMV）。@tparam T 精度。@tparam I 索引。
 * @param matrix 左矩阵。@param vector 右向量。@pre 内维度和设备一致，外部访问已同步。
 * @return 沿用矩阵策略的新向量，支持 32/64 位结构索引。
 * @code auto result = thermofem::linalg::multiply(matrix, vector); @endcode
 */
template<Scalar T, CsrIndex I> Vector<T> multiply(const CsrMatrix<T, I>& matrix, const Vector<T>& vector) {
    if (matrix.columnCount() != vector.size()) { throw std::invalid_argument("Sparse matrix-vector inner dimensions must match"); }
    detail::OperationScope scope(detail::commonDevice({matrix.rowOffsets().deviceId(), vector.deviceId()}));
    Vector<T> result(matrix.rowCount());
    if (matrix.nonZeroCount()) {
        auto a = detail::canonical<I>(matrix);
        auto x = detail::deviceCopy<T>(vector);
        detail::sparseMv(a, x.deviceData(), result.deviceData());
    } else { detail::zero(result.deviceData(), result.size()); }
    result.changeMemoryType(matrix.values().memoryStrategy());
    return result;
}
/** @brief 计算 CSR 与稠密矩阵乘积（SpMM）。@tparam T 精度。@tparam I 索引。
 * @param matrix 左 CSR。@param dense 右矩阵。@pre 内维度和设备一致，外部访问已同步。
 * @return 沿用左策略的行主序矩阵。@code auto result = thermofem::linalg::multiply(sparse, dense); @endcode
 */
template<Scalar T, CsrIndex I> DenseMatrix<T> multiply(const CsrMatrix<T, I>& matrix, const DenseMatrix<T>& dense) {
    if (matrix.columnCount() != dense.rowCount()) { throw std::invalid_argument("Sparse-dense inner dimensions must match"); }
    detail::OperationScope scope(detail::commonDevice({matrix.rowOffsets().deviceId(), dense.values().deviceId()}));
    DenseMatrix<T> result(matrix.rowCount(), dense.columnCount());
    if (matrix.nonZeroCount() && dense.columnCount()) {
        auto a = detail::canonical<I>(matrix);
        auto b = detail::deviceCopy(dense.values());
        detail::sparseMm(a, dense.columnCount(), b.deviceData(), result.deviceData());
    } else { detail::zero(result.deviceData(), result.values().size()); }
    result.changeMemoryType(matrix.values().memoryStrategy());
    return result;
}
/** @brief 计算 CSR 的 a+beta*b。@tparam T 精度。@tparam I 索引。
 * @param a 左矩阵。@param b 右矩阵。@param beta 右系数。
 * @pre 形状和设备一致，行列及非零数适配 int，外部访问已同步。
 * @return 沿用 a 策略和索引类型的规范化 CSR。
 * @code auto result = thermofem::linalg::add(a, b); @endcode
 */
template<Scalar T, CsrIndex I>
CsrMatrix<T, I> add(const CsrMatrix<T, I>& a, const CsrMatrix<T, I>& b, std::type_identity_t<T> beta = T{1}) {
    if (a.rowCount() != b.rowCount() || a.columnCount() != b.columnCount()) { throw std::invalid_argument("CSR shapes must match"); }
    detail::librarySize(a.rowCount()); detail::librarySize(a.columnCount());
    detail::librarySize(a.nonZeroCount()); detail::librarySize(b.nonZeroCount());
    detail::OperationScope scope(detail::commonDevice({a.rowOffsets().deviceId(), b.rowOffsets().deviceId()}));
    if (!a.nonZeroCount() && !b.nonZeroCount()) { return detail::emptySparse<T, I>(a.rowCount(), a.columnCount(), a.values().memoryStrategy()); }
    auto left = detail::canonical<int>(a), right = detail::canonical<int>(b);
    const auto output = detail::sparseAdd(left, right, beta);
    return detail::restoreSparse<T, I>(a.rowCount(), a.columnCount(), output, a.values().memoryStrategy());
}
/** @brief 计算 CSR 矩阵差。@tparam T 精度。@tparam I 索引。@param a 左矩阵。@param b 右矩阵。
 * @pre 同 add()。@return 规范化 CSR。@code auto result = thermofem::linalg::subtract(a, b); @endcode
 */
template<Scalar T, CsrIndex I> CsrMatrix<T, I> subtract(const CsrMatrix<T, I>& a, const CsrMatrix<T, I>& b) { return add(a, b, T{-1}); }
/** @brief 用 cuSPARSE 转置 CSR。@tparam T 精度。@tparam I 索引。@param matrix 输入。
 * @pre 行列和非零数适配 int，外部访问已同步。@return 沿用输入策略和索引类型的规范化转置 CSR。
 * @code auto result = thermofem::linalg::transpose(matrix); @endcode
 */
template<Scalar T, CsrIndex I> CsrMatrix<T, I> transpose(const CsrMatrix<T, I>& matrix) {
    detail::librarySize(matrix.rowCount()); detail::librarySize(matrix.columnCount()); detail::librarySize(matrix.nonZeroCount());
    detail::OperationScope scope(matrix.rowOffsets().deviceId());
    if (!matrix.nonZeroCount()) { return detail::emptySparse<T, I>(matrix.columnCount(), matrix.rowCount(), matrix.values().memoryStrategy()); }
    auto input = detail::canonical<int>(matrix);
    return detail::restoreSparse<T, I>(matrix.columnCount(), matrix.rowCount(), detail::sparseTranspose(input), matrix.values().memoryStrategy());
}
/** @brief 计算 CSR 矩阵乘积（SpGEMM）。@tparam T 精度。@tparam I 索引。@param a 左矩阵。@param b 右矩阵。
 * @pre 内维度和设备一致，行列、输入及输出非零数适配 int，外部访问已同步。
 * @return 沿用 a 策略和索引类型的规范化 CSR。@code auto result = thermofem::linalg::multiply(a, b); @endcode
 */
template<Scalar T, CsrIndex I> CsrMatrix<T, I> multiply(const CsrMatrix<T, I>& a, const CsrMatrix<T, I>& b) {
    if (a.columnCount() != b.rowCount()) { throw std::invalid_argument("Sparse matrix inner dimensions must match"); }
    detail::librarySize(a.rowCount()); detail::librarySize(a.columnCount()); detail::librarySize(b.columnCount());
    detail::librarySize(a.nonZeroCount()); detail::librarySize(b.nonZeroCount());
    detail::OperationScope scope(detail::commonDevice({a.rowOffsets().deviceId(), b.rowOffsets().deviceId()}));
    if (!a.nonZeroCount() || !b.nonZeroCount()) { return detail::emptySparse<T, I>(a.rowCount(), b.columnCount(), a.values().memoryStrategy()); }
    auto left = detail::canonical<int>(a), right = detail::canonical<int>(b);
    return detail::restoreSparse<T, I>(a.rowCount(), b.columnCount(), detail::sparseGemm(left, right), a.values().memoryStrategy());
}
/** @brief 计算 CSR 的 Frobenius 范数，先合并重复项。@tparam T 精度。@tparam I 索引。@param matrix 输入。
 * @pre 规范化后非零数不超过 INT_MAX，外部访问已同步。@return 矩阵范数。
 * @code auto value = thermofem::linalg::frobeniusNorm(matrix); @endcode
 */
template<Scalar T, CsrIndex I> T frobeniusNorm(const CsrMatrix<T, I>& matrix) {
    if (!matrix.nonZeroCount()) { return T{}; }
    detail::OperationScope scope(matrix.rowOffsets().deviceId());
    auto input = detail::canonical<I>(matrix);
    return detail::blasNorm(detail::librarySize(input.nonZeroCount()), input.deviceValues(), false);
}
} // 命名空间 thermofem::linalg

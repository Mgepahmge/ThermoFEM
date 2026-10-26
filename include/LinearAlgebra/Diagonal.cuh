/** @file Diagonal.cuh
 * @brief 提供独立的 CSR 逆对角线提取接口。
 */
#pragma once
#include <LinearAlgebra/Operations.cuh>
#include <cmath>
namespace thermofem::linalg {
/** @brief 提取并校验方阵的逆对角线，合并重复项。
 * @tparam T 精度。@tparam I 索引。@param matrix 输入方阵。
 * @pre 非空方阵，对角线非零、有限且倒数有限。@return global 逆对角线向量。
 * @code auto inverse = inverseDiagonal(a); @endcode
 */
template<Scalar T, CsrIndex I> Vector<T> inverseDiagonal(const CsrMatrix<T, I>& matrix) {
    if (!matrix.rowCount() || matrix.rowCount() != matrix.columnCount()) {
        throw std::invalid_argument("Inverse diagonal requires a nonempty square matrix");
    }
    detail::OperationScope scope(matrix.values().deviceId());
    const auto offsets = detail::hostCopy(matrix.rowOffsets());
    const auto columns = detail::hostCopy(matrix.columnIndices());
    const auto values = detail::hostCopy(matrix.values());
    std::vector<T> inverse(matrix.rowCount());
    for (std::size_t row = 0; row < matrix.rowCount(); ++row) {
        T diagonal{};
        for (auto j = offsets[row]; j < offsets[row+1]; ++j) {
            if (columns[j] == static_cast<I>(row)) { diagonal += values[j]; }
        }
        if (!std::isfinite(diagonal) || diagonal == 0 || !std::isfinite(T(1)/diagonal)) {
            throw std::invalid_argument("Missing or invalid diagonal");
        }
        inverse[row] = T(1)/diagonal;
    }
    return Vector<T>{std::span<const T>(inverse)};
}
} // namespace thermofem::linalg

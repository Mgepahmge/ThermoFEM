/** @file Aggregation.cuh
 * @brief 提供独立的强连接聚合、平滑延拓和 Galerkin 粗化接口。
 */
#pragma once
#include <LinearAlgebra/Operations.cuh>
#include <cmath>
#include <map>
/** @brief 提供与求解组件解耦的多重网格层次构造运算。 */
namespace thermofem::multigrid {
/** @brief 按最强可用邻居贪心配对，构造经过一次 Jacobi 平滑的延拓。
 * @tparam T 精度。@tparam I 索引。@param matrix 规范化方阵。@param weight 延拓平滑权重。@param threshold 强连接阈值。
 * @pre 各对角非零。@return 延拓矩阵，无法粗化时返回空矩阵。
 * @code auto p = prolongation(a, weight, threshold); @endcode
 */
template<linalg::Scalar T, CsrIndex I>
CsrMatrix<T, I> prolongation(const CsrMatrix<T, I>& matrix, T weight, T threshold) {
    if (!matrix.rowCount() || matrix.rowCount() != matrix.columnCount() || !std::isfinite(weight)
        || weight <= 0 || weight >= 2 || !std::isfinite(threshold) || threshold < 0 || threshold > 1) {
        throw std::invalid_argument("Invalid aggregation matrix or parameters");
    }
    const auto n = matrix.rowCount();
    const auto offsets = linalg::detail::hostCopy(matrix.rowOffsets());
    const auto columns = linalg::detail::hostCopy(matrix.columnIndices());
    const auto values = linalg::detail::hostCopy(matrix.values());
    for (const T value : values) {
        if (!std::isfinite(value)) { throw std::invalid_argument("Nonfinite aggregation input"); }
    }
    std::vector<I> aggregate(n, I(-1));
    I coarse{};
    for (std::size_t row = 0; row < n; ++row) {
        if (aggregate[row] >= 0) { continue; }
        T maximum{}, strongest{};
        I neighbor = -1;
        for (auto j = offsets[row]; j < offsets[row+1]; ++j) {
            if (columns[j] != static_cast<I>(row)) { maximum = std::max(maximum, std::abs(values[j])); }
        }
        for (auto j = offsets[row]; j < offsets[row+1]; ++j) {
            const auto column = columns[j];
            const T magnitude = std::abs(values[j]);
            if (column != static_cast<I>(row) && aggregate[column] < 0 && magnitude > strongest
                && magnitude >= threshold*maximum) {
                neighbor = column; strongest = magnitude;
            }
        }
        aggregate[row] = coarse;
        if (neighbor >= 0) { aggregate[neighbor] = coarse; }
        ++coarse;
    }
    if (static_cast<std::size_t>(coarse) >= n) { return {}; }
    std::vector<I> pOffsets{0}, pColumns;
    std::vector<T> pValues;
    for (std::size_t row = 0; row < n; ++row) {
        T diagonal{};
        for (auto j = offsets[row]; j < offsets[row+1]; ++j) {
            if (columns[j] == static_cast<I>(row)) { diagonal += values[j]; }
        }
        if (diagonal == 0 || !std::isfinite(T(1)/diagonal)) { throw std::invalid_argument("Invalid AMG diagonal"); }
        std::map<I, T> entries{{aggregate[row], 1}};
        for (auto j = offsets[row]; j < offsets[row+1]; ++j) {
            entries[aggregate[columns[j]]] -= weight * (values[j]/diagonal);
        }
        for (const auto& [column, value] : entries) {
            if (value != 0) { pColumns.push_back(column); pValues.push_back(value); }
        }
        if (!std::in_range<I>(pValues.size())) { throw std::length_error("AMG prolongation index overflow"); }
        pOffsets.push_back(static_cast<I>(pValues.size()));
    }
    return CsrMatrix<T, I>(n, coarse, pOffsets, pColumns, pValues);
}
/** @brief 计算 Galerkin 粗层矩阵 R*A*P。@tparam T 精度。@tparam I 索引。
 * @param matrix 细层矩阵。@param prolongation 延拓。@param restriction 限制。
 * @pre 维度设备匹配，CSR 结构运算满足 32 位库接口限制。
 * @return 粗层矩阵。@code auto coarse = galerkin(a, p, r); @endcode
 */
template<linalg::Scalar T, CsrIndex I>
CsrMatrix<T, I> galerkin(const CsrMatrix<T, I>& matrix, const CsrMatrix<T, I>& prolongation,
                         const CsrMatrix<T, I>& restriction) {
    if (matrix.rowCount() != matrix.columnCount() || matrix.rowCount() != prolongation.rowCount()
        || restriction.columnCount() != matrix.rowCount() || restriction.rowCount() != prolongation.columnCount()) {
        throw std::invalid_argument("Galerkin transfer shape mismatch");
    }
    return linalg::multiply(linalg::multiply(restriction, matrix), prolongation);
}
} // namespace thermofem::multigrid


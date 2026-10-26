/** @file CoarseLu.cuh
 * @brief 提供独立的小规模主机端带主元 LU 分解与缓存求解接口。
 */
#pragma once
#include <LinearAlgebra/Operations.cuh>
#include <cmath>
namespace thermofem::linalg {
/** @brief 使用部分主元 LU 求解有界的小规模粗层系统；不依赖 cuSOLVER。
 * @code CoarseLu<double> coarse(matrix); coarse.apply(rhs, x); @endcode
 */
template<linalg::Scalar T> class CoarseLu {
public:
    /** @brief 下载并分解粗矩阵。@tparam I 索引。@param matrix 粗矩阵。
     * @pre 方阵，调用者已限制维度。@return 无返回值。@code CoarseLu<double> lu(a); @endcode
     */
    template<CsrIndex I> explicit CoarseLu(const CsrMatrix<T, I>& matrix) : n_(matrix.rowCount()),
        lu_(thermofem::detail::checkedProduct(n_, n_), 0), pivots_(n_), rhs_(n_) {
        if (!n_ || n_ != matrix.columnCount()) { throw std::invalid_argument("Coarse LU requires a nonempty square matrix"); }
        const auto offsets = linalg::detail::hostCopy(matrix.rowOffsets());
        const auto columns = linalg::detail::hostCopy(matrix.columnIndices());
        const auto values = linalg::detail::hostCopy(matrix.values());
        T scale{};
        for (std::size_t row = 0; row < n_; ++row) {
            for (auto j = offsets[row]; j < offsets[row+1]; ++j) { lu_[row*n_+columns[j]] += values[j]; }
        }
        for (T value : lu_) { scale = std::max(scale, std::abs(value)); }
        if (!std::isfinite(scale) || scale == 0) { throw std::invalid_argument("Singular AMG coarse matrix"); }
        for (std::size_t k = 0; k < n_; ++k) {
            std::size_t pivot = k;
            for (std::size_t row = k+1; row < n_; ++row) {
                if (std::abs(lu_[row*n_+k]) > std::abs(lu_[pivot*n_+k])) { pivot = row; }
            }
            if (std::abs(lu_[pivot*n_+k])/scale <= std::numeric_limits<T>::epsilon()) {
                throw std::invalid_argument("Singular or numerically singular AMG coarse matrix");
            }
            pivots_[k] = pivot;
            if (pivot != k) {
                for (std::size_t j = 0; j < n_; ++j) { std::swap(lu_[k*n_+j], lu_[pivot*n_+j]); }
            }
            for (std::size_t row = k+1; row < n_; ++row) {
                lu_[row*n_+k] /= lu_[k*n_+k];
                for (std::size_t j = k+1; j < n_; ++j) { lu_[row*n_+j] -= lu_[row*n_+k]*lu_[k*n_+j]; }
            }
        }
        for (T value : lu_) {
            if (!std::isfinite(value)) { throw std::invalid_argument("Nonfinite coarse LU factors"); }
        }
    }
    /** @brief 用缓存因子求解。@param rhs 右端项。@param x 输出。
     * @pre 长度等于粗层维度且设备一致。@return 无返回值。@code coarse.apply(b, x); @endcode
     */
    void apply(const Vector<T>& rhs, Vector<T>& x) {
        if (rhs.size() != n_ || x.size() != n_) { throw std::invalid_argument("Coarse LU vector shape mismatch"); }
        detail::commonDevice({rhs.deviceId(), x.deviceId()});
        rhs.copyToHost(rhs_);
        for (std::size_t k = 0; k < n_; ++k) { std::swap(rhs_[k], rhs_[pivots_[k]]); }
        for (std::size_t row = 0; row < n_; ++row) {
            for (std::size_t j = 0; j < row; ++j) { rhs_[row] -= lu_[row*n_+j]*rhs_[j]; }
        }
        for (std::size_t row = n_; row-- > 0;) {
            for (std::size_t j = row+1; j < n_; ++j) { rhs_[row] -= lu_[row*n_+j]*rhs_[j]; }
            rhs_[row] /= lu_[row*n_+row];
        }
        x.copyFromHost(rhs_);
    }
private:
    std::size_t n_; ///< 粗层维度。
    std::vector<T> lu_; ///< 行主序 LU 因子。
    std::vector<std::size_t> pivots_; ///< 每一步的行交换。
    std::vector<T> rhs_; ///< 三角求解缓存。
};

} // namespace thermofem::linalg


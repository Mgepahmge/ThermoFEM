/** @file Jacobi.cuh
 * @brief 定义带松弛系数和固定扫描次数的 GPU Jacobi 平滑器。
 */
#pragma once
#include <Solver/Interfaces.cuh>
#include <LinearAlgebra/Diagonal.cuh>
#include <cmath>

namespace thermofem {
/** @brief 复用逆对角线和向量缓存执行加权 Jacobi；实例不可并发调用。
 * @details setup 保存矩阵快照；失败后保留原状态。输入输出允许别名，支持所有内存策略。
 * @code Jacobi<double> smoother(2, 2.0/3); smoother.setup(a); smoother.apply(b, x); @endcode
 */
template<linalg::Scalar T, CsrIndex I = int> class Jacobi final : public Smoother<T, I> {
public:
    /** @brief 指定平滑参数。@param sweeps 扫描次数。@param weight 松弛系数。
     * @pre sweeps 正，weight 有限且在 (0,2) 内。@return 无返回值。
     * @code Jacobi<double> smoother(2, 0.7); @endcode
     */
    explicit Jacobi(int sweeps = 2, T weight = T(2)/T(3)) : sweeps_(sweeps), weight_(weight) {
        if (sweeps < 1 || !std::isfinite(weight) || weight <= 0 || weight >= 2) {
            throw std::invalid_argument("Invalid Jacobi parameters");
        }
    }
    /** @brief 复制矩阵并缓存逆对角线。@param matrix 系数矩阵。
     * @pre 非空有限值方阵，合并重复项后各对角非零且倒数有限。
     * @return 无返回值。@code smoother.setup(a); @endcode
     */
    void setup(const CsrMatrix<T, I>& matrix) override {
        iteration::validateMatrix(matrix);
        linalg::detail::OperationScope scope(matrix.rowOffsets().deviceId());
        auto next = std::make_unique<State>(matrix);
        next->inverse_ = linalg::inverseDiagonal(next->context_.matrix());
        state_ = std::move(next);
    }
    /** @brief 从传入初值执行指定扫描次数。@param rhs 右端项。@param x 输入输出解。
     * @pre setup 成功，尺寸设备匹配。@return 无返回值。
     * @code smoother.apply(b, x); @endcode
     */
    void apply(const Vector<T>& rhs, Vector<T>& x) override {
        if (!state_) { throw std::logic_error("Jacobi requires setup"); }
        auto& s = *state_;
        const auto& matrix = s.context_.matrix();
        iteration::validateVectors(matrix.rowCount(), matrix.rowOffsets().deviceId(), rhs, x);
        s.rhs_.copyDataFrom(rhs); s.x_.copyDataFrom(x);
        for (int sweep = 0; sweep < sweeps_; ++sweep) {
            s.context_.multiply(s.x_, s.product_);
            s.context_.jacobi(weight_, s.inverse_, s.rhs_, s.product_, s.x_);
        }
        x.copyDataFrom(s.x_);
    }
private:
    /** @brief 拥有一次 setup 对应的所有资源。
     * @code State state(matrix); @endcode
     */
    struct State {
        linalg::IterationContext<T, I> context_; ///< 矩阵快照与持久化库资源。
        Vector<T> inverse_, rhs_, x_, product_; ///< 逆对角线及执行缓存。
        /** @brief 分配缓存。@param matrix 输入。@pre 当前设备正确。@return 无返回值。
         * @code State state(matrix); @endcode
         */
        explicit State(const CsrMatrix<T, I>& matrix) : context_(matrix), inverse_(matrix.rowCount()),
            rhs_(matrix.rowCount()), x_(matrix.rowCount()), product_(matrix.rowCount()) {}
    };
    int sweeps_; ///< 每次 apply 的扫描次数。
    T weight_; ///< 松弛系数。
    std::unique_ptr<State> state_; ///< setup 成功后建立的独占状态。
};
} // namespace thermofem

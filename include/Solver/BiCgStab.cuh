/** @file BiCgStab.cuh
 * @brief 通过独立线性代数及预条件接口实现右预条件 BiCGStab。
 */
#pragma once
#include <Preconditioner/Amg.cuh>

namespace thermofem {
/** @brief 配置求解容差、迭代上限及真实残差刷新周期。
 * @code BiCgStabOptions<double> options; options.relativeTolerance_ = 1e-10; @endcode
 */
template<linalg::Scalar T> struct BiCgStabOptions {
    int maxIterations_{1000}; ///< 最大迭代数，零表示仅检查初值。
    T relativeTolerance_{std::same_as<T, float> ? T(1e-5) : T(1e-8)}; ///< 相对右端项二范数的容差。
    T absoluteTolerance_{0}; ///< 绝对残差容差。
    int residualRefreshInterval_{50}; ///< 真实残差刷新并重启递推的间隔。
};

/** @brief 实现右预条件 BiCGStab，默认组合 AMG，可注入任意 Preconditioner。
 * @details 不直接调用 CUDA 内核或库句柄。所有向量、矩阵运算通过独立接口执行。
 * 返回前总是重算真实残差；递推残差达到阈值但真实残差未达标时重启递推。
 * 实例不可并发调用。setup 失败使本求解器失效，必须重新成功 setup。
 * @par 依赖与所有权
 * BiCgStab 是 Solver，独占一个可替换的 Preconditioner 和一组 State 工作向量。
 * 不继承预条件子、不直接访问其内部层次，也不直接发射 CUDA 内核。
 * 预条件子必须在同一次 apply 内表示固定的线性动作，否则本算法的递推假设不成立。
 * @par 数学约定
 * 使用右预条件：先计算 phat=M^-1*p，再用 v=A*phat；第二方向同样计算 shat 和 t。
 * x 始终是原方程的物理解，输入初值不作预条件变换。
 * 停止阈值为 max(atol,rtol*||rhs||2)，返回报告使用真实残差 rhs-A*x。
 * @par 可靠性与终止
 * 递推残差只用于触发候选收敛检查；实际收敛必须通过重新计算的真实残差确认。
 * 周期刷新或候选收敛校验失败后，以真实残差重置影子残差并重新开始方向递推。
 * 零分母、零 rho 或零 omega 报告 Breakdown；非有限标量报告 NonFinite。
 * 这些检测不能替代矩阵条件数分析，也不保证任意非奇异矩阵都能收敛。
 * @par 生命周期与异常
 * setup 保存矩阵快照并分配固定数量的长度为 n 的向量；外部矩阵随后可销毁。
 * 新 setup 开始即清空旧 State，以免注入预条件子 setup 失败后出现状态不一致。
 * apply 同步返回并保留输出存储；正常数值失败返回当前近似解和报告，调用或 CUDA
 * 错误通过异常传播。抛异常时不会将内部迭代结果复制回外部 x。
 * @code BiCgStab<double> solver; solver.setup(a); auto result = solver.apply(b, x); @endcode
 */
template<linalg::Scalar T, CsrIndex I = int> class BiCgStab final : public Solver<T, I> {
public:
    /** @brief 保存配置并接管预条件子。@param options 求解配置。@param preconditioner 可选预条件子。
     * @pre 迭代数非负，刷新周期正，容差有限，0<=rtol<1，atol>=0。
     * @return 无返回值。@code BiCgStab<double> solver(options, std::make_unique<Identity<double>>()); @endcode
     */
    explicit BiCgStab(BiCgStabOptions<T> options = {}, std::unique_ptr<Preconditioner<T, I>> preconditioner = {})
        : options_(options), preconditioner_(std::move(preconditioner)) {
        if (options.maxIterations_ < 0 || options.residualRefreshInterval_ < 1
            || !std::isfinite(options.relativeTolerance_) || options.relativeTolerance_ < 0 || options.relativeTolerance_ >= 1
            || !std::isfinite(options.absoluteTolerance_) || options.absoluteTolerance_ < 0) {
            throw std::invalid_argument("Invalid BiCGStab options");
        }
        if (!preconditioner_) { preconditioner_ = std::make_unique<Amg<T, I>>(); }
    }
    /** @brief 建立矩阵快照、工作向量和预条件子。@param matrix 系数矩阵。
     * @pre 非空有限值方阵，并满足所选预条件子的条件。@return 无返回值。
     * @code solver.setup(a); @endcode
     * @details 顺序为：撤销就绪状态、检查方阵、建立执行上下文和缓存、初始化预条件子、
     * 发布新状态。通过后两次 apply 可复用同一矩阵状态而求解不同右端项。
     * @throws std::invalid_argument 输入或预条件条件不满足。
     * @note 任一步失败都要求重新成功 setup；不承诺保留上一次可用状态。
     */
    void setup(const CsrMatrix<T, I>& matrix) override {
        state_.reset();
        iteration::validateMatrix(matrix);
        linalg::detail::OperationScope scope(matrix.rowOffsets().deviceId());
        auto next = std::make_unique<State>(matrix);
        preconditioner_->setup(next->context_.matrix());
        state_ = std::move(next);
    }
    /** @brief 从给定初值求解，保留输出策略，允许 rhs 与 x 别名。@param rhs 右端项。
     * @param x 输入输出近似解。@pre setup 成功，尺寸设备匹配。
     * @return 收敛、达到上限、递推崩溃或非有限数状态及真实残差。
     * @code auto result = solver.apply(b, x); @endcode
     * @details rhs 和 x 先被分别暂存，支持两者别名。初值满足真实残差阈值时返回零次迭代。
     * MaxIterations、Breakdown、NonFinite 都是正常报告，不应仅凭函数返回就认定求解成功。
     * 零右端项时 relativeResidual_ 定义为绝对残差；通常应提供可达到的绝对容差。
     * @throws std::logic_error 尚未成功 setup 或最近一次 setup 失败。
     * @throws std::invalid_argument 向量尺寸或设备不匹配。
     */
    SolveResult<T> apply(const Vector<T>& rhs, Vector<T>& x) override {
        if (!state_) { throw std::logic_error("BiCGStab requires setup"); }
        auto& s = *state_;
        iteration::validateVectors(s.x_.size(), s.x_.deviceId(), rhs, x);
        s.rhs_.copyDataFrom(rhs); s.x_.copyDataFrom(x);
        const auto report = solve();
        x.copyDataFrom(s.x_);
        return report;
    }
private:
    /** @brief 保存矩阵上下文与固定工作向量。
     * @code State state(matrix); @endcode
     */
    struct State {
        linalg::IterationContext<T, I> context_; ///< 系数矩阵及持久化执行上下文。
        Vector<T> rhs_, x_; ///< 输入快照与当前解。
        Vector<T> r_, shadow_, p_, v_, s_, t_; ///< BiCGStab 递推向量。
        Vector<T> preconditionedP_, preconditionedS_; ///< 预条件后的搜索方向。
        /** @brief 分配全部缓存。@param matrix 系数矩阵。@pre 当前设备正确。
         * @return 无返回值。@code State state(matrix); @endcode
         */
        explicit State(const CsrMatrix<T, I>& matrix) : context_(matrix), rhs_(matrix.rowCount()), x_(matrix.rowCount()),
            r_(matrix.rowCount()), shadow_(matrix.rowCount()), p_(matrix.rowCount()), v_(matrix.rowCount()),
            s_(matrix.rowCount()), t_(matrix.rowCount()), preconditionedP_(matrix.rowCount()), preconditionedS_(matrix.rowCount()) {}
    };
    /** @brief 重算真实残差并形成终止报告。@param status 原因。@param iterations 迭代次数。
     * @param normB 右端项范数。@param tolerance 停止阈值。
     * @pre 状态完整。@return 校验后的报告。@code return finish(status, k, normB, tolerance); @endcode
     */
    SolveResult<T> finish(SolveStatus status, int iterations, T normB, T tolerance) {
        auto& s = *state_;
        s.context_.residual(s.rhs_, s.x_, s.r_);
        const T residual = s.context_.norm(s.r_);
        if (!std::isfinite(residual) || !std::isfinite(normB)) { status = SolveStatus::NonFinite; }
        else if (residual <= tolerance) { status = SolveStatus::Converged; }
        else if (status == SolveStatus::Converged) { status = SolveStatus::MaxIterations; }
        return {status, iterations, residual, normB > 0 ? residual/normB : residual};
    }
    /** @brief 在已分配工作向量上执行递推，无参数。@pre 状态完整。
     * @return 真实残差报告。@code auto result = solve(); @endcode
     * @details fresh 标记首次方向或重启方向，避免使用前一轮无效的 p、v、rhoPrevious。
     * 每轮先更新 alpha 方向；若中间残差已达阈值，跳过 omega 方向并校验真实残差。
     * 否则执行第二次预条件和 SpMV，计算 omega 后更新解与残差。
     * rhoPrevious 只在下一轮使用已有方向时参与递推。
     */
    SolveResult<T> solve() {
        auto& s = *state_;
        auto& operations = s.context_;
        const T normB = operations.norm(s.rhs_);
        const T tolerance = std::max(options_.absoluteTolerance_, options_.relativeTolerance_*normB);
        operations.residual(s.rhs_, s.x_, s.r_);
        T residual = operations.norm(s.r_);
        if (!std::isfinite(normB) || !std::isfinite(residual)) { return finish(SolveStatus::NonFinite, 0, normB, tolerance); }
        if (residual <= tolerance) { return finish(SolveStatus::Converged, 0, normB, tolerance); }
        s.shadow_.copyDataFrom(s.r_);
        T rhoPrevious = 1, alpha = 1, omega = 1;
        bool fresh = true;
        for (int k = 0; k < options_.maxIterations_; ++k) {
            const int iterations = k+1;
            const T rho = operations.dot(s.shadow_, s.r_);
            if (!std::isfinite(rho)) { return finish(SolveStatus::NonFinite, iterations, normB, tolerance); }
            if (rho == 0) { return finish(SolveStatus::Breakdown, iterations, normB, tolerance); }
            if (fresh) { s.p_.copyDataFrom(s.r_); fresh = false; }
            else {
                const T beta = (rho/rhoPrevious)*(alpha/omega);
                if (!std::isfinite(beta)) { return finish(SolveStatus::NonFinite, iterations, normB, tolerance); }
                operations.axpy(-omega, s.v_, s.p_);
                operations.scale(beta, s.p_);
                operations.axpy(T(1), s.r_, s.p_);
            }
            preconditioner_->apply(s.p_, s.preconditionedP_);
            operations.multiply(s.preconditionedP_, s.v_);
            const T denominator = operations.dot(s.shadow_, s.v_);
            if (!std::isfinite(denominator)) { return finish(SolveStatus::NonFinite, iterations, normB, tolerance); }
            if (denominator == 0) { return finish(SolveStatus::Breakdown, iterations, normB, tolerance); }
            alpha = rho/denominator;
            if (!std::isfinite(alpha)) { return finish(SolveStatus::NonFinite, iterations, normB, tolerance); }
            s.s_.copyDataFrom(s.r_);
            operations.axpy(-alpha, s.v_, s.s_);
            operations.axpy(alpha, s.preconditionedP_, s.x_);
            const T normS = operations.norm(s.s_);
            if (!std::isfinite(normS)) { return finish(SolveStatus::NonFinite, iterations, normB, tolerance); }
            if (normS <= tolerance) {
                operations.residual(s.rhs_, s.x_, s.r_);
                residual = operations.norm(s.r_);
                if (residual <= tolerance || !std::isfinite(residual)) {
                    return finish(SolveStatus::MaxIterations, iterations, normB, tolerance);
                }
                s.shadow_.copyDataFrom(s.r_); fresh = true;
                continue;
            }
            preconditioner_->apply(s.s_, s.preconditionedS_);
            operations.multiply(s.preconditionedS_, s.t_);
            const T tt = operations.dot(s.t_, s.t_), ts = operations.dot(s.t_, s.s_);
            if (!std::isfinite(tt) || !std::isfinite(ts)) { return finish(SolveStatus::NonFinite, iterations, normB, tolerance); }
            if (tt == 0) { return finish(SolveStatus::Breakdown, iterations, normB, tolerance); }
            omega = ts/tt;
            if (!std::isfinite(omega)) { return finish(SolveStatus::NonFinite, iterations, normB, tolerance); }
            if (omega == 0) { return finish(SolveStatus::Breakdown, iterations, normB, tolerance); }
            operations.axpy(omega, s.preconditionedS_, s.x_);
            s.r_.copyDataFrom(s.s_); operations.axpy(-omega, s.t_, s.r_);
            residual = operations.norm(s.r_);
            if (!std::isfinite(residual)) { return finish(SolveStatus::NonFinite, iterations, normB, tolerance); }
            if (residual <= tolerance || iterations % options_.residualRefreshInterval_ == 0) {
                operations.residual(s.rhs_, s.x_, s.r_);
                residual = operations.norm(s.r_);
                if (residual <= tolerance || !std::isfinite(residual)) {
                    return finish(SolveStatus::MaxIterations, iterations, normB, tolerance);
                }
                s.shadow_.copyDataFrom(s.r_); fresh = true;
            }
            rhoPrevious = rho;
        }
        return finish(SolveStatus::MaxIterations, options_.maxIterations_, normB, tolerance);
    }
    BiCgStabOptions<T> options_; ///< 固定求解配置。
    std::unique_ptr<Preconditioner<T, I>> preconditioner_; ///< 独立的可替换预条件组件。
    std::unique_ptr<State> state_; ///< 当前 setup 对应的状态。
};
} // namespace thermofem

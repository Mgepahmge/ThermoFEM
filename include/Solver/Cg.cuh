/** @file Cg.cuh
 * @brief 通过独立线性代数及预条件接口实现预条件共轭梯度法。
 */
#pragma once
#include <Preconditioner/Identity.cuh>

namespace thermofem {
    /** @brief 配置 CG 容差、迭代上限和真实残差刷新周期。
     * @code CgOptions<double> options; options.relativeTolerance_ = 1e-10; @endcode */
    template <linalg::Scalar T>
    struct CgOptions {
        int maxIterations_{1000}; ///< 最大迭代数，零表示仅检查初值。
        T relativeTolerance_{std::same_as<T, float> ? T(1e-5) : T(1e-8)}; ///< 相对右端项范数的容差。
        T absoluteTolerance_{0}; ///< 绝对残差容差。
        int residualRefreshInterval_{50}; ///< 真实残差刷新并重启方向的周期，必须为正。
    };

    /** @brief 组合固定预条件动作的 CG 求解器，默认使用恒等预条件子。
     * @tparam T float 或 double。@tparam I CSR 索引类型。
     * @par 数学契约
     * 系数矩阵必须对称正定，预条件动作必须固定、线性且对称正定。
     * setup 不执行全矩阵对称性或正定性认证；运行中检测到 r^T*z 或 p^T*A*p
     * 非正时报告 Breakdown，但未触发检测也不能证明输入满足上述契约。
     * 恒等预条件对应普通 CG；注入相容预条件子后为 PCG。AMG 的相容性不在本组件中保证。
     * @par 迭代与收敛
     * z=M^-1*r，p=z+beta*p，alpha=(r^T*z)/(p^T*A*p)。全部运算调用
     * IterationContext 和 Preconditioner，不直接调用 CUDA 内核或 cuSOLVER。
     * 阈值为 max(atol,rtol*||rhs||2)。候选收敛及周期刷新均重算真实残差；
     * 未收敛时以真实残差重新开始方向递推，避免递推残差漂移导致误报。
     * 所有正常返回报告真实残差；零右端项的相对残差定义为绝对残差。
     * @par 所有权与异常保证
     * setup 持有矩阵快照和固定工作向量，失败后必须重新成功 setup。
     * apply 保存 rhs 和初值，允许二者别名，保留输出地址和内存策略。
     * 正常数值失败返回当前近似解；异常传播时不将内部解复制回输出。
     * 方法同步返回，实例不可并发使用。预条件子由求解器独占，不依赖其具体类型。
     * @code Cg<double> solver; solver.setup(a); auto report = solver.apply(b, x); @endcode */
    template <linalg::Scalar T, CsrIndex I = int>
    class Cg final : public Solver<T, I> {
    public:
        /** @brief 保存配置并接管预条件子，空指针表示恒等预条件。
         * @param options 求解配置。@param preconditioner 可选的相容预条件子。
         * @pre 迭代数非负，刷新周期正，容差有限，0<=rtol<1，atol>=0。
         * @note 无返回值。@code Cg<double> solver(options, std::make_unique<Identity<double>>()); @endcode */
        explicit Cg(CgOptions<T> options = {}, std::unique_ptr<Preconditioner<T, I>> preconditioner = {}) :
            options_(options), preconditioner_(std::move(preconditioner)) {
            if (options.maxIterations_ < 0 || options.residualRefreshInterval_ < 1 ||
                !std::isfinite(options.relativeTolerance_) || options.relativeTolerance_ < 0 ||
                options.relativeTolerance_ >= 1 || !std::isfinite(options.absoluteTolerance_) ||
                options.absoluteTolerance_ < 0) {
                throw std::invalid_argument("Invalid CG options");
            }
            if (!preconditioner_) {
                preconditioner_ = std::make_unique<Identity<T, I>>();
            }
        }
        /** @brief 建立矩阵快照、运算上下文及预条件状态。
         * @param matrix 非空有限值对称正定方阵。
         * @pre 设备可用，满足预条件子要求；正定性由调用方保证。
         * @note 无返回值；失败撤销旧就绪状态。@code solver.setup(a); @endcode */
        void setup(const CsrMatrix<T, I>& matrix) override {
            state_.reset();
            iteration::validateMatrix(matrix);
            linalg::detail::OperationScope scope(matrix.rowOffsets().deviceId());
            auto next = std::make_unique<State>(matrix);
            preconditioner_->setup(next->context_.matrix());
            state_ = std::move(next);
        }
        /** @brief 从传入初值求解，允许右端项与输出别名。
         * @param rhs 右端项。@param x 输入初值并接收解。
         * @pre setup 成功，向量尺寸设备匹配，矩阵和预条件子满足 CG 数学契约。
         * @return 状态、迭代次数及真实残差；初值收敛时迭代次数为零。
         * @throws std::logic_error 尚未成功 setup。
         * @throws std::invalid_argument 尺寸或设备不匹配。
         * @code auto report = solver.apply(b, x); @endcode */
        SolveResult<T> apply(const Vector<T>& rhs, Vector<T>& x) override {
            if (!state_) {
                throw std::logic_error("CG requires setup");
            }
            auto& s = *state_;
            iteration::validateVectors(s.x_.size(), s.x_.deviceId(), rhs, x);
            s.rhs_.copyDataFrom(rhs);
            s.x_.copyDataFrom(x);
            const auto report = solve();
            x.copyDataFrom(s.x_);
            return report;
        }

    private:
        /** @brief 持有矩阵快照和 CG 工作区。@code State state(matrix); @endcode */
        struct State {
            linalg::IterationContext<T, I> context_; ///< 矩阵快照和可复用库资源。
            Vector<T> rhs_, x_; ///< 右端项快照和当前解。
            Vector<T> r_, z_, p_, product_; ///< 残差、预条件残差、搜索方向及矩阵乘积。
            /** @brief 在当前设备分配缓存。@param matrix 系数矩阵。
             * @pre 当前设备正确。@note 无返回值。@code State state(matrix); @endcode */
            explicit State(const CsrMatrix<T, I>& matrix) :
                context_(matrix), rhs_(matrix.rowCount()), x_(matrix.rowCount()), r_(matrix.rowCount()),
                z_(matrix.rowCount()), p_(matrix.rowCount()), product_(matrix.rowCount()) {}
        };
        /** @brief 重算真实残差并形成报告。
         * @param status 原始终止原因。@param iterations 迭代次数。
         * @param normB 右端项范数。@param tolerance 停止阈值。
         * @pre 状态完整。@return 校验后的终止报告。
         * @code return finish(status, k, normB, tolerance); @endcode */
        SolveResult<T> finish(SolveStatus status, int iterations, T normB, T tolerance) {
            auto& s = *state_;
            s.context_.residual(s.rhs_, s.x_, s.r_);
            const T residual = s.context_.norm(s.r_);
            if (!std::isfinite(residual) || !std::isfinite(normB)) {
                status = SolveStatus::NonFinite;
            }
            else if (residual <= tolerance) {
                status = SolveStatus::Converged;
            }
            else if (status == SolveStatus::Converged) {
                status = SolveStatus::MaxIterations;
            }
            return {status, iterations, residual, normB > 0 ? residual / normB : residual};
        }
        /** @brief 在持久工作区上执行 PCG 递推，无参数。
         * @pre 状态完整且数学契约满足。@return 使用真实残差的报告。
         * @code auto report = solve(); @endcode */
        SolveResult<T> solve() {
            auto& s = *state_;
            auto& op = s.context_;
            const T normB = op.norm(s.rhs_);
            const T tolerance = std::max(options_.absoluteTolerance_, options_.relativeTolerance_ * normB);
            op.residual(s.rhs_, s.x_, s.r_);
            T residual = op.norm(s.r_);
            if (!std::isfinite(normB) || !std::isfinite(residual)) {
                return finish(SolveStatus::NonFinite, 0, normB, tolerance);
            }
            if (residual <= tolerance) {
                return finish(SolveStatus::Converged, 0, normB, tolerance);
            }
            T previousRho = 0;
            bool fresh = true;
            for (int k = 0; k < options_.maxIterations_; ++k) {
                const int iterations = k + 1;
                preconditioner_->apply(s.r_, s.z_);
                const T rho = op.dot(s.r_, s.z_);
                if (!std::isfinite(rho)) {
                    return finish(SolveStatus::NonFinite, iterations, normB, tolerance);
                }
                if (rho <= 0) {
                    return finish(SolveStatus::Breakdown, iterations, normB, tolerance);
                }
                if (fresh) {
                    s.p_.copyDataFrom(s.z_);
                    fresh = false;
                }
                else {
                    const T beta = rho / previousRho;
                    if (!std::isfinite(beta)) {
                        return finish(SolveStatus::NonFinite, iterations, normB, tolerance);
                    }
                    op.scale(beta, s.p_);
                    op.axpy(T(1), s.z_, s.p_);
                }
                op.multiply(s.p_, s.product_);
                const T curvature = op.dot(s.p_, s.product_);
                if (!std::isfinite(curvature)) {
                    return finish(SolveStatus::NonFinite, iterations, normB, tolerance);
                }
                if (curvature <= 0) {
                    return finish(SolveStatus::Breakdown, iterations, normB, tolerance);
                }
                const T alpha = rho / curvature;
                if (!std::isfinite(alpha)) {
                    return finish(SolveStatus::NonFinite, iterations, normB, tolerance);
                }
                op.axpy(alpha, s.p_, s.x_);
                op.axpy(-alpha, s.product_, s.r_);
                residual = op.norm(s.r_);
                if (!std::isfinite(residual)) {
                    return finish(SolveStatus::NonFinite, iterations, normB, tolerance);
                }
                if (residual <= tolerance || iterations % options_.residualRefreshInterval_ == 0) {
                    op.residual(s.rhs_, s.x_, s.r_);
                    residual = op.norm(s.r_);
                    if (!std::isfinite(residual) || residual <= tolerance) {
                        return finish(SolveStatus::MaxIterations, iterations, normB, tolerance);
                    }
                    fresh = true;
                }
                previousRho = rho;
            }
            return finish(SolveStatus::MaxIterations, options_.maxIterations_, normB, tolerance);
        }
        CgOptions<T> options_; ///< 固定求解配置。
        std::unique_ptr<Preconditioner<T, I>> preconditioner_; ///< 独占可注入预条件组件。
        std::unique_ptr<State> state_; ///< 当前成功 setup 对应的状态。
    };
} // namespace thermofem

/** @file Jacobi.cuh
 * @brief 定义带松弛系数和固定扫描次数的 GPU Jacobi 平滑器。
 */
#pragma once
#include <LinearAlgebra/Diagonal.cuh>
#include <Solver/Interfaces.cuh>
#include <cmath>

namespace thermofem {
    /** @brief 复用逆对角线和向量缓存执行加权 Jacobi；实例不可并发调用。
     * @details setup 保存矩阵快照；失败后保留原状态。输入输出允许别名，支持所有内存策略。
     * @par 算法与职责
     * 每次扫描计算 product=A*x，然后执行 x+=weight*D^-1*(rhs-product)。
     * 矩阵向量积和逐项更新由独立线性代数接口实现，平滑器仅组合扫描流程。
     * 默认两次扫描、权重 2/3；它降低误差中的高频分量，不提供收敛判据。
     * 对一般矩阵，合法的权重区间不等于收敛保证。
     * @par 状态与所有权
     * State 独占矩阵上下文、逆对角线和四个长度相关缓存。
     * setup 先建立完整的新状态再替换旧状态，来源矩阵可随后修改或销毁。
     * apply 先暂存 rhs 和初值，再迭代并将结果复制回 x，因此允许 rhs 与 x 相同。
     * @par 同步与失败
     * apply 同步返回，保留 x 的地址和策略；工作区使用矩阵设备上的 global 存储。
     * 缺失或无效对角线在 setup 抛异常。执行期间 CUDA 错误向上传播；实例不可并发使用。
     * @code Jacobi<double> smoother(2, 2.0/3); smoother.setup(a); smoother.apply(b, x); @endcode
     */
    template <linalg::Scalar T, CsrIndex I = int>
    class Jacobi final : public Smoother<T, I> {
    public:
        /** @brief 指定平滑参数。@param sweeps 扫描次数。@param weight 松弛系数。
         * @pre sweeps 正，weight 有限且在 (0,2) 内。@return 无返回值。
         * @code Jacobi<double> smoother(2, 0.7); @endcode
         */
        explicit Jacobi(int sweeps = 2, T weight = T(2) / T(3)) : sweeps_(sweeps), weight_(weight) {
            if (sweeps < 1 || !std::isfinite(weight) || weight <= 0 || weight >= 2) {
                throw std::invalid_argument("Invalid Jacobi parameters");
            }
        }
        /** @brief 复制矩阵并缓存逆对角线。@param matrix 系数矩阵。
         * @pre 非空有限值方阵，合并重复项后各对角非零且倒数有限。
         * @return 无返回值。@code smoother.setup(a); @endcode
         * @details 先规范化 CSR，再通过 inverseDiagonal 提取逆对角线。重复列被合并，
         * 因而抵消为零的重复对角项同样被拒绝。成功后不再引用输入矩阵。
         * @throws std::invalid_argument 方阵或对角线条件不满足。
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
         * @details 不将初值重置为零。内部缓存承接各类内存策略，循环内复用已有库资源。
         * 空间开销由 setup 决定，扫描过程不重新建立矩阵描述符。
         * @throws std::logic_error 尚未成功 setup。
         * @throws std::invalid_argument 向量尺寸或设备不匹配。
         */
        void apply(const Vector<T>& rhs, Vector<T>& x) override {
            if (!state_) {
                throw std::logic_error("Jacobi requires setup");
            }
            auto& s = *state_;
            const auto& matrix = s.context_.matrix();
            iteration::validateVectors(matrix.rowCount(), matrix.rowOffsets().deviceId(), rhs, x);
            s.rhs_.copyDataFrom(rhs);
            s.x_.copyDataFrom(x);
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
            explicit State(const CsrMatrix<T, I>& matrix) :
                context_(matrix), inverse_(matrix.rowCount()), rhs_(matrix.rowCount()), x_(matrix.rowCount()),
                product_(matrix.rowCount()) {}
        };
        int sweeps_; ///< 每次 apply 的扫描次数。
        T weight_; ///< 松弛系数。
        std::unique_ptr<State> state_; ///< setup 成功后建立的独占状态。
    };
} // namespace thermofem

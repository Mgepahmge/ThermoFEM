/** @file Amg.cuh
 * @brief 实现平滑聚合 AMG、Galerkin 层次、V-cycle 和小规模粗层 LU。
 */
#pragma once
#include <Smoother/Jacobi.cuh>
#include <functional>
#include <LinearAlgebra/CoarseLu.cuh>
#include <Multigrid/Aggregation.cuh>

namespace thermofem {
/** @brief 配置平滑聚合层次与资源上限。
 * @code AmgOptions<double> options; options.coarseRows_ = 16; @endcode
 */
template<linalg::Scalar T> struct AmgOptions {
    int maxLevels_{20}; ///< 最大层数，包含最细层。
    int coarseRows_{16}; ///< 达到此维度后停止粗化。
    int maxDenseCoarseRows_{256}; ///< CPU 粗层稠密 LU 的硬上限。
    int smoothingSweeps_{2}; ///< 默认 Jacobi 每次前、后平滑的扫描次数。
    T jacobiWeight_{T(2)/T(3)}; ///< 默认平滑器及延拓平滑的权重。
    T strengthThreshold_{T(0.25)}; ///< 强连接阈值，相对本行最大非对角绝对值。
};



/** @brief 组合独立平滑器、平滑聚合层次和粗层求解的 AMG V-cycle 预条件子。
 * @details setup 在主机构造聚合与延拓，cuSPARSE 计算 R=P^T、Ac=R*A*P。
 * apply 复用 GPU 工作区；小规模粗层使用 CPU LU，需要两次主机设备传输。
 * 主要面向椭圆型问题，不保证任意非对称矩阵的收敛；实例不可并发调用。
 * @par 组件关系
 * Amg 是 Preconditioner，组合每层的 Smoother、矩阵执行上下文和传输上下文。
 * 工厂每次创建独立平滑器，不共享可变缓存；默认实现为加权 Jacobi。
 * 聚合、Galerkin 乘积和粗层 LU 分别属于 Multigrid 与 LinearAlgebra 运算层。
 * @par 层次构造
 * 强连接阈值按本行最大非对角绝对值定义，贪心配对形成分片常数 P0。
 * 一次 Jacobi 平滑得到 P=(I-weight*D^-1*A)*P0，再取 R=P^T、Ac=R*A*P。
 * 达到目标维度、层数上限或无法继续聚合时停止；最后检查粗层硬上限并分解。
 * @par 状态与资源
 * setup 以临时 State 构建完整层次，成功后替换旧状态，失败保留旧状态。
 * 每层持有 rhs、x、residual、correction 及各上下文的资源；存储取决于全部层的
 * 维度与非零数之和。平滑延拓可能增加非零数，本实现尚无限制算子复杂度的截断策略。
 * @par 适用边界
 * 固定扫描次数、固定传输和固定粗层因子构成供 BiCGStab 使用的固定线性动作。
 * 注入平滑器也必须保持这一性质。非对称、不定或强各向异性问题需另行验证。
 * 主机聚合与有界 CPU LU 是初版取舍，不包含参考项目中的融合和 Graph 优化。
 * @code Amg<double> amg; amg.setup(a); amg.apply(b, z); @endcode
 */
template<linalg::Scalar T, CsrIndex I = int> class Amg final : public Preconditioner<T, I> {
public:
    using SmootherFactory = std::function<std::unique_ptr<Smoother<T, I>>() >; ///< 每层平滑器工厂。
    /** @brief 保存配置与可选平滑器工厂。@param options 层次配置。@param factory 工厂，空则使用 Jacobi。
     * @pre 上限和扫描次数为正，阈值在 [0,1]，权重在 (0,2)。
     * @return 无返回值。@code Amg<double> amg(options); @endcode
     */
    explicit Amg(AmgOptions<T> options = {}, SmootherFactory factory = {}) : options_(options), factory_(std::move(factory)) {
        if (options.maxLevels_ < 1 || options.coarseRows_ < 1 || options.maxDenseCoarseRows_ < options.coarseRows_
            || options.smoothingSweeps_ < 1 || !std::isfinite(options.strengthThreshold_)
            || options.strengthThreshold_ < 0 || options.strengthThreshold_ > 1
            || !std::isfinite(options.jacobiWeight_) || options.jacobiWeight_ <= 0 || options.jacobiWeight_ >= 2) {
            throw std::invalid_argument("Invalid AMG options");
        }
        if (!factory_) {
            factory_ = [options]() { return std::make_unique<Jacobi<T, I>>(options.smoothingSweeps_, options.jacobiWeight_); };
        }
    }
    /** @brief 构建层次、平滑器与缓存，失败保留旧状态。@param matrix 最细层系数矩阵。
     * @pre 有限值非空方阵，平滑层对角有效且粗层可分解。@return 无返回值。
     * @code amg.setup(a); @endcode
     * @details 最细矩阵规范化后逐层构造传输和 Galerkin 矩阵；平滑器仅分配在非粗层。
     * 单层情形直接执行粗层 LU，因而粗层允许零对角但要求可通过主元交换分解。
     * @throws std::invalid_argument 配置不适用、工厂返回空对象、粗层过大或数值奇异。
     * @throws std::length_error 稀疏结构超出底层 32 位结构运算范围。
     */
    void setup(const CsrMatrix<T, I>& matrix) override {
        iteration::validateMatrix(matrix);
        linalg::detail::OperationScope scope(matrix.rowOffsets().deviceId());
        auto next = std::make_unique<State>();
        CsrMatrix<T, I> current = linalg::detail::canonical<I>(matrix);
        for (int depth = 0; ; ++depth) {
            auto level = std::make_unique<Level>(current);
            const auto& a = level->context_.matrix();
            if (a.rowCount() <= static_cast<std::size_t>(options_.coarseRows_) || depth+1 == options_.maxLevels_) {
                next->levels_.push_back(std::move(level)); break;
            }
            auto p = multigrid::prolongation(a, options_.jacobiWeight_, options_.strengthThreshold_);
            if (!p.rowCount()) { next->levels_.push_back(std::move(level)); break; }
            auto r = linalg::transpose(p);
            current = multigrid::galerkin(a, p, r);
            level->prolongation_ = std::make_unique<linalg::IterationContext<T, I>>(p);
            level->restriction_ = std::make_unique<linalg::IterationContext<T, I>>(r);
            level->smoother_ = factory_();
            if (!level->smoother_) { throw std::invalid_argument("AMG smoother factory returned null"); }
            level->smoother_->setup(a);
            next->levels_.push_back(std::move(level));
        }
        const auto& coarse = next->levels_.back()->context_.matrix();
        if (coarse.rowCount() > static_cast<std::size_t>(options_.maxDenseCoarseRows_)) {
            throw std::invalid_argument("AMG failed to reach bounded coarse size; adjust hierarchy options");
        }
        next->coarse_ = std::make_unique<linalg::CoarseLu<T>>(coarse);
        state_ = std::move(next);
    }
    /** @brief 从零执行一个 V-cycle，覆盖输出。@param rhs 右端项。@param x 输出。
     * @pre setup 成功，尺寸设备匹配。@return 无返回值。
     * @code amg.apply(b, z); @endcode
     * @details 保存 rhs 后，从零执行前平滑、残差限制、粗层递归、延拓修正、后平滑。
     * 每次调用覆盖 x，不读取其旧值；允许输入输出别名并保留输出策略和地址。
     * 返回值是近似逆作用，不代表 A*x=rhs 已满足任何求解容差。
     * @throws std::logic_error 尚未成功 setup。
     * @throws std::invalid_argument 向量尺寸或设备不匹配。
     */
    void apply(const Vector<T>& rhs, Vector<T>& x) override {
        if (!state_) { throw std::logic_error("AMG requires setup"); }
        auto& first = *state_->levels_.front();
        iteration::validateVectors(first.x_.size(), first.x_.deviceId(), rhs, x);
        linalg::detail::OperationScope scope(first.x_.deviceId());
        first.rhs_.copyDataFrom(rhs);
        cycle(0);
        x.copyDataFrom(first.x_);
    }
    /** @brief 查询已建立层数，无参数、无额外约束。@return 层数，未 setup 时为零。
     * @code auto levels = amg.levelCount(); @endcode
     */
    std::size_t levelCount() const noexcept { return state_ ? state_->levels_.size() : 0; }
private:
    /** @brief 拥有单层运算、平滑和传输缓存。
     * @code Level level(matrix); @endcode
     */
    struct Level {
        linalg::IterationContext<T, I> context_; ///< 本层矩阵执行上下文。
        Vector<T> rhs_, x_, residual_, correction_; ///< 右端、解、残差、延拓修正缓存。
        std::unique_ptr<Smoother<T, I>> smoother_; ///< 通过独立接口使用平滑器。
        std::unique_ptr<linalg::IterationContext<T, I>> prolongation_, restriction_; ///< 传输运算上下文。
        /** @brief 分配本层缓存。@param matrix 系数矩阵。@pre 当前设备正确。
         * @return 无返回值。@code Level level(a); @endcode
         */
        explicit Level(const CsrMatrix<T, I>& matrix) : context_(matrix), rhs_(matrix.rowCount()), x_(matrix.rowCount()),
            residual_(matrix.rowCount()), correction_(matrix.rowCount()) {}
    };
    /** @brief 独占完整层次。
     * @code State state; @endcode
     */
    struct State {
        std::vector<std::unique_ptr<Level>> levels_; ///< 从细到粗的层次。
        std::unique_ptr<linalg::CoarseLu<T>> coarse_; ///< 最粗层分解。
    };
    /** @brief 递归执行零初值 V-cycle。@param depth 当前层。
     * @pre 状态完整，当前设备正确。@return 无返回值。@code cycle(0); @endcode
     * @details 最粗层直接用缓存 LU；其他层按前平滑、r=b-A*x、bc=R*r、
     * ec=Vcycle(bc)、x+=P*ec、后平滑依次组合独立运算。
     * 每层 x 在进入时重置，因此上次 apply 的缓存内容不影响本次动作。
     */
    void cycle(std::size_t depth) {
        auto& level = *state_->levels_[depth];
        if (depth+1 == state_->levels_.size()) { state_->coarse_->apply(level.rhs_, level.x_); return; }
        level.context_.clear(level.x_);
        level.smoother_->apply(level.rhs_, level.x_);
        level.context_.residual(level.rhs_, level.x_, level.residual_);
        auto& coarse = *state_->levels_[depth+1];
        level.restriction_->multiply(level.residual_, coarse.rhs_);
        cycle(depth+1);
        level.prolongation_->multiply(coarse.x_, level.correction_);
        level.context_.axpy(T(1), level.correction_, level.x_);
        level.smoother_->apply(level.rhs_, level.x_);
    }
    AmgOptions<T> options_; ///< 不可变层次配置。
    SmootherFactory factory_; ///< 可替换平滑器构造方式。
    std::unique_ptr<State> state_; ///< 当前成功建立的层次。
};
} // namespace thermofem

/** @file Amg.cuh
 * @brief 实现平滑聚合 AMG、Galerkin 层次、V-cycle 和小规模粗层 LU。
 */
#pragma once
#include <Smoother/Jacobi.cuh>
#include <functional>
#include <LinearAlgebra/CoarseLu.cuh>
#include <Multigrid/Aggregation.cuh>
#include <numeric>

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
     * @pre 有限值非空方阵，各层对角有效且粗层可分解。@return 无返回值。
     * @code amg.setup(a); @endcode
     */
    void setup(const CsrMatrix<T, I>& matrix) override {
        iteration::validateMatrix(matrix);
        linalg::detail::OperationScope scope(matrix.values().deviceId());
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
     */
    void cycle(std::size_t depth) {
        auto& level = *state_->levels_[depth];
        if (depth+1 == state_->levels_.size()) { state_->coarse_->apply(level.rhs_, level.x_); return; }
        linalg::detail::zero(level.x_.deviceData(), level.x_.size());
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

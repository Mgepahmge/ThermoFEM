/** @file Amg.cuh
 * @brief 组合 GPU 原生聚合、独立平滑器、粗层求解和 V/F/W 调度的 AMG。
 */
#pragma once
#include <Multigrid/Cycle.cuh>
#include <Multigrid/GpuAggregation.cuh>
#include <Smoother/Jacobi.cuh>
#include <functional>
#include <string>
namespace thermofem {
    /** @brief 配置 YHAMG-GPU 原生聚合层次和循环。
     * @code AmgOptions<double> o; o.cycle_=multigrid::Cycle::W; @endcode */
    template <linalg::Scalar T>
    struct AmgOptions {
        int maxLevels_{20}; ///< 最大层数，1..20。
        int coarseRows_{64}; ///< 目标粗阶数，1..64。
        int matchingRounds_{8}; ///< 每阶段同步匹配轮数，1..8。
        int preSweeps_{1}, postSweeps_{1}; ///< 前后平滑次数，可为零。
        int coarseSweeps_{8}; ///< LU 不可用时的固定 Jacobi 次数。
        T jacobiWeight_{T(0.75)}; ///< 默认 Jacobi 权重。
        T strengthThreshold_{T(0.25)}; ///< 相对行最大非对角阈值。
        bool absorbSingletons_{true}; ///< 两轮单点吸收。
        bool compactSortKeys_{false}; ///< 使用紧凑坐标键。
        multigrid::Cycle cycle_{multigrid::Cycle::V}; ///< 循环类型。
    };
    /** @brief 每层规模和聚合统计。@code auto n=amg.levelStatistics()[0].rows_; @endcode */
    struct AmgLevelStatistics {
        std::size_t rows_{}, nonzeros_{}; ///< 本层维度与非零数。
        std::array<int, 4> aggregateSizes_{}; ///< 一到四成员聚合数。
    };
    /** @brief 组合独立 GPU 运算的非平滑聚合 AMG 预条件子。
     * @tparam T float/double。@tparam I 32/64 位索引。
     * @par 算法与组件
     * 使用用户非公开 YHAMG-GPU 原生路径：GPU 双向强图、两阶段同步互选匹配与图收缩、
     * 两轮单点吸收、分片常数 P、直接归并 Ac=P^T*A*P。每聚合最多四点，粗矩阵非零数
     * 不增加。不使用平滑延拓、截断、SpGEMM 或 cuSOLVER。GpuAggregation 负责独立
     * setup 运算，AggregateTransfer 拥有传输表，Cycle 只调度。Amg 组合这些元件与
     * 每层独立 Smoother，不继承存储类型。
     * @par 粗层和停止
     * 所有层验证有限数值、非零对角和有限倒数。目标阶数、最大层数、空强图或不能收缩
     * 均可终止。粗层 <=64 阶使用 GPU 部分主元 LU，阈值与参考一致；奇异 LU 或较大
     * 终止层采用固定 coarseSweeps 次 Jacobi，保持参考的 fallback 行为。
     * @par 循环与初值
     * V 单次误差递归；W 对同一粗右端迭代两次且第二次保留第一次解；F 先限制原右端
     * 递归生成初值、延拓覆盖、再作 V 修正，与参考 F 语义一致。apply 从零开始，允许
     * rhs/x 别名，保持输出策略和地址。固定层次与固定次数构成线性近似逆动作。
     * @par 生命周期与执行
     * setup 完整构建临时状态，成功才替换，失败保留旧状态。apply 复用缓存，不可并发。
     * 原生 setup 使用 double/int；float/宽索引在 GPU 边界转换，结构受 int32 容量限制。
     * 迭代计算使用 T/I。融合内核与 CUDA Graph 未绑定到组件，因此不承诺参考原版性能；
     * 此处保留完整数值阶段，并用解耦运算支持三种调度。
     * @code AmgOptions<double> o; o.cycle_=multigrid::Cycle::F; Amg<double> amg(o); amg.setup(a); amg.apply(b,z);
     * @endcode */
    template <linalg::Scalar T, CsrIndex I = int>
    class Amg final : public Preconditioner<T, I> {
    public:
        using SmootherFactory = std::function<std::unique_ptr<Smoother<T, I>>()>; ///< 独立组件工厂。
        /** @brief 保存配置及可选工厂。
         * @param options 配置。@param factory 分别为前后平滑创建组件。
         * @pre 配置合法；注入平滑器提供固定线性动作。@note 无返回值。
         * @details 默认使用独立 Jacobi，扫描数取 pre/postSweeps；注入工厂时扫描数由
         * 返回组件定义。对应 sweeps 为零仍跳过阶段。粗层 fallback 不使用工厂。
         * @code Amg<double> amg(options); @endcode */
        explicit Amg(AmgOptions<T> options = {}, SmootherFactory factory = {}) :
            options_(options), factory_(std::move(factory)) {
            if (options.maxLevels_ < 1 || options.maxLevels_ > 20 || options.coarseRows_ < 1 ||
                options.coarseRows_ > 64 || options.matchingRounds_ < 1 || options.matchingRounds_ > 8 ||
                options.preSweeps_ < 0 || options.postSweeps_ < 0 || options.coarseSweeps_ < 1 ||
                !std::isfinite(options.strengthThreshold_) || options.strengthThreshold_ < 0 ||
                options.strengthThreshold_ > 1 || !std::isfinite(options.jacobiWeight_) || options.jacobiWeight_ <= 0 ||
                options.jacobiWeight_ >= 2 ||
                (options.cycle_ != multigrid::Cycle::V && options.cycle_ != multigrid::Cycle::W &&
                 options.cycle_ != multigrid::Cycle::F)) {
                throw std::invalid_argument("Invalid AMG options");
            }
        }
        /** @brief 构建层次、因子、组件和缓存，失败保留旧状态。
         * @param matrix 输入方阵。@pre 非空有限，各层对角有效，原生图容量可表示。
         * @note 无返回值。@throws std::invalid_argument 非法数值、对角或空工厂。
         * @throws std::length_error 原生图容量不足。@code amg.setup(a); @endcode */
        void setup(const CsrMatrix<T, I>& matrix) override {
            iteration::validateMatrix(matrix);
            linalg::detail::OperationScope scope(matrix.rowOffsets().deviceId());
            auto next = std::make_unique<State>();
            multigrid::GpuAggregation operations(options_.compactSortKeys_);
            auto global = matrix;
            global.changeMemoryType(GlobalMemory{});
            auto native = operations.canonical(multigrid::toNativeMatrix(global));
            for (int depth = 0;; ++depth) {
                operations.validate(native);
                auto typed = multigrid::fromNativeMatrix<T, I>(native);
                // 迭代精度下也须存在有限倒数，防止 double 转 float 产生非法对角。
                (void)linalg::inverseDiagonal(typed);
                auto level = std::make_unique<Level>(typed);
                next->statistics_.push_back({native.rowCount(), native.nonZeroCount(), {}});
                if (native.rowCount() <= size_t(options_.coarseRows_)) {
                    next->stopReason_ = "coarse size";
                    next->levels_.push_back(std::move(level));
                    break;
                }
                if (depth + 1 == options_.maxLevels_) {
                    next->stopReason_ = "level limit";
                    next->levels_.push_back(std::move(level));
                    break;
                }
                auto graph = operations.strength(native, double(options_.strengthThreshold_));
                if (!graph.nonZeroCount()) {
                    next->stopReason_ = "empty strength graph";
                    next->levels_.push_back(std::move(level));
                    break;
                }
                auto transfer = operations.aggregate(graph, options_.matchingRounds_, options_.absorbSingletons_);
                if (transfer.coarseRows_ >= transfer.fineRows_) {
                    next->stopReason_ = "no contraction";
                    next->levels_.push_back(std::move(level));
                    break;
                }
                auto coarse = operations.galerkin(native, transfer);
                next->statistics_.back().aggregateSizes_ = transfer.histogram_;
                level->transfer_ = std::make_unique<multigrid::AggregateTransfer>(std::move(transfer));
                if (options_.preSweeps_) {
                    level->pre_ = makeSmoother(typed, options_.preSweeps_);
                }
                if (options_.postSweeps_) {
                    level->post_ = makeSmoother(typed, options_.postSweeps_);
                }
                next->levels_.push_back(std::move(level));
                native = std::move(coarse);
            }
            if (native.rowCount() <= 64) {
                next->factor_ = multigrid::factorCoarse(native);
            }
            if (next->factor_.factored_) {
                next->coarseSolver_ = "pivoted GPU LU";
            }
            else {
                next->coarseSolver_ = next->factor_.singular_ ? "Jacobi (singular LU)" : "Jacobi (coarsening stopped)";
                next->fallback_ = std::make_unique<Jacobi<T, I>>(options_.coarseSweeps_, options_.jacobiWeight_);
                next->fallback_->setup(next->levels_.back()->context_.matrix());
            }
            next->visits_.resize(next->levels_.size());
            state_ = std::move(next);
        }
        /** @brief 从零执行配置的 V/W/F 循环，覆盖输出。
         * @param rhs 右端。@param x 输出，可与 rhs 别名。
         * @pre setup 成功，尺寸设备匹配，支持全部内存策略。@note 无返回值。
         * @details 先保存 rhs；调度逐层调用独立平滑、残差、限制、延拓及粗层求解接口。
         * 不判断收敛容差，调用方求解器负责真实残差验证。@code amg.apply(b,z); @endcode */
        void apply(const Vector<T>& rhs, Vector<T>& x) override {
            if (!state_) {
                throw std::logic_error("AMG requires setup");
            }
            auto& first = *state_->levels_.front();
            iteration::validateVectors(first.x_.size(), first.x_.deviceId(), rhs, x);
            linalg::detail::OperationScope scope(first.x_.deviceId());
            first.rhs_.copyDataFrom(rhs);
            std::fill(state_->visits_.begin(), state_->visits_.end(), 0);
            CycleAdapter adapter{*state_};
            multigrid::executeCycle(options_.cycle_, 0, adapter, true);
            x.copyDataFrom(first.x_);
        }
        /** @brief 查询层数，无参数、无约束。@return 层数。@code amg.levelCount(); @endcode */
        std::size_t levelCount() const noexcept { return state_ ? state_->levels_.size() : 0; }
        /** @brief 查询层统计，无参数。@pre setup 成功。@return 只读统计。@code amg.levelStatistics(); @endcode */
        const std::vector<AmgLevelStatistics>& levelStatistics() const { return state().statistics_; }
        /** @brief 查询停止原因，无参数。@pre setup 成功。@return 描述。@code amg.stopReason(); @endcode */
        const std::string& stopReason() const { return state().stopReason_; }
        /** @brief 查询粗层路径，无参数。@pre setup 成功。@return 描述。@code amg.coarseSolver(); @endcode */
        const std::string& coarseSolver() const { return state().coarseSolver_; }
        /** @brief 查询最近 apply 的每层访问数，无参数。@pre setup 成功。
         * @return 计数。@code amg.lastCycleVisits(); @endcode */
        const std::vector<std::size_t>& lastCycleVisits() const { return state().visits_; }

    private:
        /** @brief 组合一层运算、独立前后平滑器及传输缓存。@code Level level(a); @endcode */
        struct Level {
            linalg::IterationContext<T, I> context_; ///< 本层矩阵运算。
            Vector<T> rhs_, x_, residual_; ///< 右端、近似解、残差缓存。
            std::unique_ptr<Smoother<T, I>> pre_, post_; ///< 独立前后平滑器。
            std::unique_ptr<multigrid::AggregateTransfer> transfer_; ///< 到下一层的传输。
            /** @brief 分配层缓存。@param a 矩阵。@pre 当前设备正确。@note 无返回值。@code Level l(a); @endcode
             */
            explicit Level(const CsrMatrix<T, I>& a) :
                context_(a), rhs_(a.rowCount()), x_(a.rowCount()), residual_(a.rowCount()) {}
        };
        /** @brief 完整层次的事务性所有者。@code State state; @endcode */
        struct State {
            std::vector<std::unique_ptr<Level>> levels_; ///< 从细到粗层次。
            std::vector<AmgLevelStatistics> statistics_; ///< 层规模与聚合统计。
            std::vector<std::size_t> visits_; ///< 最近循环访问数。
            multigrid::CoarseFactor factor_; ///< GPU 粗层因子。
            std::unique_ptr<Smoother<T, I>> fallback_; ///< 粗层固定 Jacobi。
            std::string stopReason_, coarseSolver_; ///< 终止与粗层路径。
        };
        /** @brief 非拥有适配器，将独立层运算交给循环调度。
         * @code CycleAdapter adapter{state}; multigrid::executeCycle(multigrid::Cycle::V,0,adapter); @endcode */
        struct CycleAdapter {
            State& state_; ///< apply 期间的有效状态。
            /** @brief 判定粗层。@param d 层号。@pre 层号有效。@return 是否最粗层。@code adapter.isCoarsest(0);
             * @endcode */
            bool isCoarsest(std::size_t d) const { return d + 1 == state_.levels_.size(); }
            /** @brief 记录访问。@param d 层号。@pre 层号有效。@note 无返回值。@code adapter.begin(0); @endcode
             */
            void begin(std::size_t d) { ++state_.visits_[d]; }
            /** @brief 清零初值。@param d 层号。@pre 层号有效。@note 无返回值。@code adapter.clear(0); @endcode
             */
            void clear(std::size_t d) {
                auto& l = *state_.levels_[d];
                l.context_.clear(l.x_);
            }
            /** @brief 调用独立粗层 LU 或保留初值的固定 Jacobi。
             * @param d 最粗层。@pre 状态完整。@note 无返回值。@code adapter.coarseSolve(d); @endcode */
            void coarseSolve(std::size_t d) {
                auto& l = *state_.levels_[d];
                if (state_.factor_.factored_) {
                    multigrid::solveCoarse(state_.factor_, l.rhs_, l.x_);
                }
                else {
                    state_.fallback_->apply(l.rhs_, l.x_);
                }
            }
            /** @brief 前平滑。@param d 非粗层。@pre 状态完整。@note 无返回值。@code adapter.preSmooth(0);
             * @endcode */
            void preSmooth(std::size_t d) {
                auto& l = *state_.levels_[d];
                if (l.pre_) {
                    l.pre_->apply(l.rhs_, l.x_);
                }
            }
            /** @brief 后平滑。@param d 非粗层。@pre 状态完整。@note 无返回值。@code adapter.postSmooth(0);
             * @endcode */
            void postSmooth(std::size_t d) {
                auto& l = *state_.levels_[d];
                if (l.post_) {
                    l.post_->apply(l.rhs_, l.x_);
                }
            }
            /** @brief F 嵌套初值所需的原右端限制。
             * @param d 非粗层。@pre 状态完整。@note 无返回值。@code adapter.restrictRhs(0); @endcode */
            void restrictRhs(std::size_t d) {
                auto& l = *state_.levels_[d];
                multigrid::restrictAggregates(*l.transfer_, l.rhs_, state_.levels_[d + 1]->rhs_);
            }
            /** @brief 分别调用残差和限制，构造粗层误差右端。
             * @param d 非粗层。@pre 状态完整。@note 无返回值。@code adapter.restrictResidual(0); @endcode */
            void restrictResidual(std::size_t d) {
                auto& l = *state_.levels_[d];
                l.context_.residual(l.rhs_, l.x_, l.residual_);
                multigrid::restrictAggregates(*l.transfer_, l.residual_, state_.levels_[d + 1]->rhs_);
            }
            /** @brief 延拓累加粗解，F 覆盖初值前由调度器先清零。
             * @param d 非粗层。@pre 状态完整。@note 无返回值。@code adapter.prolong(0); @endcode */
            void prolong(std::size_t d) {
                auto& l = *state_.levels_[d];
                multigrid::prolongAggregates(*l.transfer_, state_.levels_[d + 1]->x_, l.x_);
            }
        };
        /** @brief 创建并建立平滑器。@param a 矩阵。@param sweeps 默认次数。
         * @pre 当前设备正确。@return 独立组件。@code auto s=makeSmoother(a,1); @endcode */
        std::unique_ptr<Smoother<T, I>> makeSmoother(const CsrMatrix<T, I>& a, int sweeps) {
            auto smoother = factory_ ? factory_() : std::make_unique<Jacobi<T, I>>(sweeps, options_.jacobiWeight_);
            if (!smoother) {
                throw std::invalid_argument("AMG smoother factory returned null");
            }
            smoother->setup(a);
            return smoother;
        }
        /** @brief 获取已建立状态，无参数。@pre setup 成功。@return 状态。@code state(); @endcode */
        const State& state() const {
            if (!state_) {
                throw std::logic_error("AMG requires setup");
            }
            return *state_;
        }
        AmgOptions<T> options_; ///< 固定配置。
        SmootherFactory factory_; ///< 可注入平滑器工厂。
        std::unique_ptr<State> state_; ///< 最近成功状态。
    };
} // namespace thermofem

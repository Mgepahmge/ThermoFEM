/** @file amg_native_test.cpp
 * @brief 独立主机算法对照 GPU 聚合阶段，并验证 V/W/F 的调度与数值动作。
 */
#include <Multigrid/GpuAggregation.cuh>
#include <Preconditioner/Amg.cuh>
#include <Solver/BiCgStab.cuh>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include "support/NativeAggregationOracle.cuh"
/** @brief 原生 AMG 算法与主机数学基准测试。 */
namespace {
    using namespace thermofem;
    /** @brief CUDA 测试环境。@code NativeAmgTest fixture; @endcode */
    class NativeAmgTest : public ::testing::Test {
    protected:
        /** @brief 选择 GPU，无参数、无额外约束。@return 无返回值。@code SetUp(); @endcode */
        void SetUp() override {
            int count{};
            if (cudaGetDeviceCount(&count) != cudaSuccess || !count) {
                GTEST_SKIP();
            }
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        }
    };
    /** @brief 构造带等权边、非对称边和孤立点的可重现稀疏方阵。
     * @param n 阶数。@pre n>0。@return 严格对角占优矩阵。@code auto a=sample(61); @endcode */
    io::HostCsr sample(int n) {
        io::HostCsr a;
        a.rows_ = a.columns_ = n;
        a.offsets_ = {0};
        std::mt19937 random(173 + n);
        for (int i = 0; i < n; ++i) {
            std::vector<std::pair<int, double>> row;
            double diagonal = 2;
            for (int j = 0; j < n; ++j) {
                if (i != j && i % 13 != 0 && j % 13 != 0 && (random() % 11 == 0 || std::abs(i - j) == 1)) {
                    const double value = -double(1 + random() % 4);
                    row.emplace_back(j, value);
                    diagonal -= value;
                }
            }
            row.emplace_back(i, diagonal);
            std::sort(row.begin(), row.end());
            for (auto [j, value] : row) {
                a.indices_.push_back(j);
                a.values_.push_back(value);
            }
            a.offsets_.push_back(int(a.values_.size()));
        }
        return a;
    }
    /** @brief 比较 GPU CSR 和独立主机基准。
     * @param a GPU 矩阵。@param expected 主机矩阵。@pre 无额外约束。
     * @return 无返回值。@code compare(a,expected); @endcode */
    void compare(const CsrMatrix<double>& a, const io::HostCsr& expected) {
        EXPECT_EQ(a.rowCount(), size_t(expected.rows_));
        EXPECT_EQ(linalg::detail::hostCopy(a.rowOffsets()), expected.offsets_);
        EXPECT_EQ(linalg::detail::hostCopy(a.columnIndices()), expected.indices_);
        auto values = linalg::detail::hostCopy(a.values());
        ASSERT_EQ(values.size(), expected.values_.size());
        for (size_t j = 0; j < values.size(); ++j) {
            EXPECT_NEAR(values[j], expected.values_[j], 1e-12);
        }
    }
    TEST_F(NativeAmgTest, gpuStagesMatchIndependentHostAlgorithm) {
        bool absorptionExercised = false;
        for (int n : {1, 2, 4, 17, 61, 128}) {
            for (bool compact : {false, true}) {
                const auto host = sample(n);
                CsrMatrix<double> input(n, n, host.offsets_, host.indices_, host.values_);
                multigrid::GpuAggregation operations(compact);
                auto a = operations.canonical(input);
                compare(a, testoracle::canonicalMatrix(host));
                for (double threshold : {0.0, 0.25, 1.0}) {
                    auto graph = operations.strength(a, threshold);
                    auto hostGraph = testoracle::strengthGraph(host, threshold);
                    compare(graph, hostGraph.edges_);
                    for (int rounds : {1, 8}) {
                        for (bool absorb : {false, true}) {
                            const auto expected = testoracle::aggregate(hostGraph, rounds, absorb);
                            const auto t = operations.aggregate(graph, rounds, absorb);
                            EXPECT_EQ(linalg::detail::hostCopy(t.aggregate_), expected.fineToCoarse_);
                            EXPECT_EQ(linalg::detail::hostCopy(t.offsets_), expected.offsets_);
                            EXPECT_EQ(linalg::detail::hostCopy(t.members_), expected.members_);
                            for (size_t k = 0; k < 4; ++k) {
                                EXPECT_EQ(size_t(t.histogram_[k]), expected.histogram_[k]);
                            }
                            if (absorb) {
                                const auto initial = testoracle::aggregate(hostGraph, rounds, false);
                                absorptionExercised |= initial.coarseRows() != expected.coarseRows();
                            }
                            const auto coarse = operations.galerkin(a, t);
                            compare(coarse, testoracle::aggregateGalerkin(host, expected));
                            EXPECT_LE(coarse.nonZeroCount(), a.nonZeroCount());
                            const std::vector<double> values(n, 1);
                            Vector<double> fine{std::span<const double>(values)}, restricted(t.coarseRows_);
                            multigrid::restrictAggregates(t, fine, restricted);
                            auto sums = linalg::detail::hostCopy(restricted);
                            for (int c = 0; c < t.coarseRows_; ++c) {
                                EXPECT_EQ(sums[c], expected.offsets_[c + 1] - expected.offsets_[c]);
                            }
                            multigrid::prolongAggregates(t, restricted, fine);
                            auto actual = linalg::detail::hostCopy(fine);
                            for (int i = 0; i < n; ++i) {
                                EXPECT_EQ(actual[i], 1 + sums[expected.fineToCoarse_[i]]);
                            }
                        }
                    }
                }
            }
        }
        EXPECT_TRUE(absorptionExercised);
    }
    TEST_F(NativeAmgTest, canonicalDuplicatesAndDeviceCsrValidation) {
        const std::vector<int> rp{0, 3, 6}, ci{1, 0, 0, 1, 0, 1};
        const std::vector<double> values{-1, 1, 1, 1, -1, 1};
        CsrMatrix<double> a(2, 2, rp, ci, values);
        multigrid::GpuAggregation operations;
        const auto canonical = operations.canonical(a);
        EXPECT_EQ(linalg::detail::hostCopy(canonical.values()), (std::vector<double>{2, -1, -1, 2}));
        const std::vector<int> badOffsets{0, 7, 6};
        CudaBuffer<int> offsets{std::span<const int>(badOffsets)}, indices{std::span<const int>(ci)};
        CudaBuffer<double> entries{std::span<const double>(values)};
        EXPECT_THROW(CsrMatrix<double>::fromBuffers(2, 2, std::move(offsets), std::move(indices), std::move(entries)),
                     std::invalid_argument);
        EXPECT_EQ(offsets.size(), 3u);
        EXPECT_EQ(entries.size(), 6u);
        EXPECT_THROW(operations.strength(a, -.1), std::invalid_argument);
        const auto graph = operations.strength(canonical, .25);
        EXPECT_THROW(operations.aggregate(graph, 0, true), std::invalid_argument);
    }
    TEST_F(NativeAmgTest, gpuCoarsePivotScaleAndSingularDetection) {
        const std::vector<int> rp{0, 2, 4}, ci{0, 1, 0, 1};
        for (double scale : {1e-100, 1.0, 1e100}) {
            const std::vector<double> v{scale, 2 * scale, 4 * scale, 3 * scale}, b{5 * scale, 10 * scale};
            CsrMatrix<double> a(2, 2, rp, ci, v);
            auto factor = multigrid::factorCoarse(a);
            ASSERT_TRUE(factor.factored_);
            Vector<double> rhs{std::span<const double>(b)}, x(2);
            multigrid::solveCoarse(factor, rhs, x);
            const auto actual = linalg::detail::hostCopy(x);
            EXPECT_NEAR(actual[0], 1, 1e-12);
            EXPECT_NEAR(actual[1], 2, 1e-12);
            multigrid::solveCoarse(factor, rhs, rhs);
            EXPECT_EQ(linalg::detail::hostCopy(rhs), actual);
        }
        const std::vector<double> singular{1, 1, 1, 1};
        CsrMatrix<double> a(2, 2, rp, ci, singular);
        const auto factor = multigrid::factorCoarse(a);
        EXPECT_TRUE(factor.singular_);
        EXPECT_FALSE(factor.factored_);
    }
    /** @brief 不使用循环调度器的主机数值基准，固定 Jacobi 和直接消元。
     * @details 从稀疏主机矩阵构造层次，显式实现参考的 V/W/F 递归，独立验证 GPU 调度与初值。
     * @code HostCycle oracle(matrix); auto expected=oracle.apply(multigrid::Cycle::W,b); @endcode */
    class HostCycle {
        std::vector<io::HostCsr> matrices_; ///< 主机层次。
        std::vector<testoracle::AggregateMap> maps_; ///< 聚合表。
        std::vector<std::vector<double>> rhs_, x_; ///< 每层右端与近似解。
        /** @brief 主机 Jacobi 一次更新。@param d 层。@pre 状态有效。@return 无返回值。@code smooth(0); @endcode
         */
        void smooth(size_t d) {
            const auto& a = matrices_[d];
            auto next = x_[d];
            for (int i = 0; i < a.rows_; ++i) {
                double product = 0, diag = 0;
                for (int j = a.offsets_[i]; j < a.offsets_[i + 1]; ++j) {
                    product += a.values_[j] * x_[d][a.indices_[j]];
                    if (a.indices_[j] == i) {
                        diag = a.values_[j];
                    }
                }
                next[i] += .75 * (rhs_[d][i] - product) / diag;
            }
            x_[d] = std::move(next);
        }
        /** @brief 限制右端或残差。@param d 层。@param residual 是否残差。
         * @pre 层非粗层。@return 无返回值。@code restrict(0,true); @endcode */
        void restrict(size_t d, bool residual) {
            auto fine = rhs_[d];
            const auto& a = matrices_[d];
            if (residual) {
                for (int i = 0; i < a.rows_; ++i) {
                    for (int j = a.offsets_[i]; j < a.offsets_[i + 1]; ++j) {
                        fine[i] -= a.values_[j] * x_[d][a.indices_[j]];
                    }
                }
            }
            std::fill(rhs_[d + 1].begin(), rhs_[d + 1].end(), 0);
            for (int i = 0; i < a.rows_; ++i) {
                rhs_[d + 1][maps_[d].fineToCoarse_[i]] += fine[i];
            }
        }
        /** @brief 累加延拓。@param d 层。@pre 非粗层。@return 无返回值。@code prolong(0); @endcode */
        void prolong(size_t d) {
            for (size_t i = 0; i < x_[d].size(); ++i) {
                x_[d][i] += x_[d + 1][maps_[d].fineToCoarse_[i]];
            }
        }
        /** @brief 主机部分主元消元直接解。@param d 粗层。@pre 非奇异。@return 无返回值。@code solve(d); @endcode
         */
        void solve(size_t d) {
            const auto& a = matrices_[d];
            int n = a.rows_;
            std::vector<double> m(n * n, 0), b = rhs_[d];
            for (int i = 0; i < n; ++i) {
                for (int j = a.offsets_[i]; j < a.offsets_[i + 1]; ++j) {
                    m[i * n + a.indices_[j]] = a.values_[j];
                }
            }
            for (int k = 0; k < n; ++k) {
                int p = k;
                for (int i = k + 1; i < n; ++i) {
                    if (std::abs(m[i * n + k]) > std::abs(m[p * n + k])) {
                        p = i;
                    }
                }
                for (int j = 0; j < n; ++j) {
                    std::swap(m[k * n + j], m[p * n + j]);
                }
                std::swap(b[k], b[p]);
                for (int i = k + 1; i < n; ++i) {
                    double f = m[i * n + k] / m[k * n + k];
                    for (int j = k; j < n; ++j) {
                        m[i * n + j] -= f * m[k * n + j];
                    }
                    b[i] -= f * b[k];
                }
            }
            for (int i = n - 1; i >= 0; --i) {
                double v = b[i];
                for (int j = i + 1; j < n; ++j) {
                    v -= m[i * n + j] * x_[d][j];
                }
                x_[d][i] = v / m[i * n + i];
            }
        }
        /** @brief 独立递归 V 或 W。@param d 层。@param w 是否 W。@param zero 是否清零。
         * @pre 状态完整。@return 无返回值。@code vw(0,true,true); @endcode */
        void vw(size_t d, bool w, bool zero) {
            if (zero) {
                std::fill(x_[d].begin(), x_[d].end(), 0);
            }
            if (d + 1 == matrices_.size()) {
                solve(d);
                return;
            }
            smooth(d);
            restrict(d, true);
            vw(d + 1, w, true);
            if (w) {
                vw(d + 1, w, false);
            }
            prolong(d);
            smooth(d);
        }
        /** @brief 独立实现参考嵌套 F。@param d 层。@pre 状态完整。
         * @return 无返回值。@code f(0); @endcode */
        void f(size_t d) {
            std::fill(x_[d].begin(), x_[d].end(), 0);
            if (d + 1 == matrices_.size()) {
                solve(d);
                return;
            }
            restrict(d, false);
            f(d + 1);
            prolong(d);
            vw(d, false, false);
        }

    public:
        /** @brief 构造主机层次。@param a 最细矩阵。@pre 可粗化到 <=4 阶。
         * @return 无返回值。@code HostCycle oracle(a); @endcode */
        explicit HostCycle(io::HostCsr a) {
            for (;;) {
                matrices_.push_back(a);
                rhs_.emplace_back(a.rows_, 0);
                x_.emplace_back(a.rows_, 0);
                if (a.rows_ <= 4) {
                    break;
                }
                auto t = testoracle::aggregate(testoracle::strengthGraph(a, .25), 8, true);
                a = testoracle::aggregateGalerkin(a, t);
                maps_.push_back(std::move(t));
            }
        }
        /** @brief 求主机循环动作。@param kind 类型。@param b 右端。@pre 维度正确。
         * @return 近似解。@code auto x=oracle.apply(kind,b); @endcode */
        std::vector<double> apply(multigrid::Cycle kind, const std::vector<double>& b) {
            rhs_[0] = b;
            if (kind == multigrid::Cycle::F) {
                f(0);
            }
            else {
                vw(0, kind == multigrid::Cycle::W, true);
            }
            return x_[0];
        }
    };
    /** @brief 构造一维 Poisson 主机矩阵。@param n 阶数。@pre 正数。
     * @return CSR。@code auto a=poissonHost(127); @endcode */
    io::HostCsr poissonHost(int n) {
        io::HostCsr a;
        a.rows_ = a.columns_ = n;
        a.offsets_ = {0};
        for (int i = 0; i < n; ++i) {
            if (i) {
                a.indices_.push_back(i - 1);
                a.values_.push_back(-1);
            }
            a.indices_.push_back(i);
            a.values_.push_back(2);
            if (i + 1 < n) {
                a.indices_.push_back(i + 1);
                a.values_.push_back(-1);
            }
            a.offsets_.push_back(int(a.values_.size()));
        }
        return a;
    }
    /** @brief 比较三种 GPU 循环和独立主机递归，并验证线性、别名及求解。
     * @pre GPU 可用。@return 无返回值。@code checkCycles<double,int>(); @endcode */
    template <linalg::Scalar T, CsrIndex I>
    void checkCycles() {
        const auto host = poissonHost(127);
        std::vector<I> rp(host.offsets_.begin(), host.offsets_.end()), ci(host.indices_.begin(), host.indices_.end());
        std::vector<T> av(host.values_.begin(), host.values_.end()), b(127), c(127), sum(127);
        std::vector<double> bd(127);
        for (int i = 0; i < 127; ++i) {
            b[i] = T(std::sin(.13 * i) + .4);
            bd[i] = b[i];
            c[i] = T(std::cos(.07 * i));
            sum[i] = b[i] + c[i];
        }
        CsrMatrix<T, I> a(127, 127, rp, ci, av);
        HostCycle oracle(host);
        for (auto kind : {multigrid::Cycle::V, multigrid::Cycle::W, multigrid::Cycle::F}) {
            AmgOptions<T> options;
            options.coarseRows_ = 4;
            options.cycle_ = kind;
            Amg<T, I> amg(options);
            amg.setup(a);
            ASSERT_GE(amg.levelCount(), 3u);
            Vector<T> rhs{std::span<const T>(b)}, x(127), other(127), combined(127);
            amg.apply(rhs, x);
            const auto expected = oracle.apply(kind, bd), actual = linalg::detail::hostCopy(x);
            const double tol = std::same_as<T, float> ? 2e-3 : 2e-10;
            for (int i = 0; i < 127; ++i) {
                EXPECT_NEAR(actual[i], expected[i], tol);
            }
            if (kind == multigrid::Cycle::V) {
                for (auto visits : amg.lastCycleVisits()) {
                    EXPECT_EQ(visits, 1u);
                }
            }
            if (kind == multigrid::Cycle::W) {
                size_t expectedVisits = 1;
                for (auto visits : amg.lastCycleVisits()) {
                    EXPECT_EQ(visits, expectedVisits);
                    expectedVisits *= 2;
                }
            }
            if (kind == multigrid::Cycle::F) {
                const auto& visits = amg.lastCycleVisits();
                for (size_t d = 0; d < visits.size(); ++d) {
                    EXPECT_EQ(visits[d], d + 1 == visits.size() ? visits.size() : d + 2);
                }
            }
            rhs.copyFromHost(c);
            amg.apply(rhs, other);
            rhs.copyFromHost(sum);
            amg.apply(rhs, combined);
            const auto xc = linalg::detail::hostCopy(other), xs = linalg::detail::hostCopy(combined);
            for (int i = 0; i < 127; ++i) {
                EXPECT_NEAR(xs[i], actual[i] + xc[i], tol);
            }
            rhs.copyFromHost(b);
            amg.apply(rhs, rhs);
            const auto alias = linalg::detail::hostCopy(rhs);
            for (int i = 0; i < 127; ++i) {
                EXPECT_NEAR(alias[i], actual[i], tol);
            }
            BiCgStabOptions<T> solveOptions;
            solveOptions.relativeTolerance_ = std::same_as<T, float> ? T(5e-4) : T(1e-9);
            BiCgStab<T, I> solver(solveOptions, std::make_unique<Amg<T, I>>(options));
            solver.setup(a);
            rhs.copyFromHost(b);
            std::vector<T> zero(127, 0);
            x.copyFromHost(zero);
            const auto report = solver.apply(rhs, x);
            EXPECT_EQ(report.status_, SolveStatus::Converged);
            EXPECT_LE(report.relativeResidual_, solveOptions.relativeTolerance_);
        }
    }
    TEST_F(NativeAmgTest, cyclesMatchIndependentHostDouble) { checkCycles<double, int>(); }
    TEST_F(NativeAmgTest, cyclesMatchIndependentHostFloatWide) { checkCycles<float, std::int64_t>(); }
    TEST_F(NativeAmgTest, fallbackMatchesFixedJacobiAndZeroSweeps) {
        const auto host = poissonHost(100);
        CsrMatrix<double> a(100, 100, host.offsets_, host.indices_, host.values_);
        std::vector<double> b(100, 1), zero(100, 0);
        Vector<double> rhs{std::span<const double>(b)}, x(100), expected{std::span<const double>(zero)};
        AmgOptions<double> o;
        o.maxLevels_ = 1;
        o.coarseSweeps_ = 8;
        Amg<double> amg(o);
        amg.setup(a);
        amg.apply(rhs, x);
        Jacobi<double> jacobi(8, .75);
        jacobi.setup(a);
        jacobi.apply(rhs, expected);
        EXPECT_EQ(linalg::detail::hostCopy(x), linalg::detail::hostCopy(expected));
        o.maxLevels_ = 20;
        o.coarseRows_ = 4;
        o.preSweeps_ = o.postSweeps_ = 0;
        for (auto kind : {multigrid::Cycle::V, multigrid::Cycle::W, multigrid::Cycle::F}) {
            o.cycle_ = kind;
            Amg<double> noSmoothing(o);
            noSmoothing.setup(a);
            noSmoothing.apply(rhs, x);
            auto first = linalg::detail::hostCopy(x);
            noSmoothing.apply(rhs, x);
            EXPECT_EQ(first, linalg::detail::hostCopy(x));
        }
    }
} // namespace

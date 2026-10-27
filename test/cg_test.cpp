/** @file cg_test.cpp
 * @brief 使用恒等预条件子验证 CG 的解析解、真实残差和失败契约。
 */
#include <Solver/Cg.cuh>
#include <gtest/gtest.h>
#include <limits>

/** @brief CG 的独立数值与接口测试。 */
namespace {
    using namespace thermofem;
    /** @brief 激活 CUDA 的多精度测试夹具。@code CgTest<double> fixture; @endcode */
    template <linalg::Scalar T>
    class CgTest : public ::testing::Test {
    protected:
        /** @brief 激活设备，无参数、无额外约束。@note 无返回值。@code SetUp(); @endcode */
        void SetUp() override {
            int count{};
            if (cudaGetDeviceCount(&count) != cudaSuccess || !count) {
                GTEST_SKIP() << "CUDA unavailable";
            }
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        }
    };
    using Precisions = ::testing::Types<float, double>;
    TYPED_TEST_SUITE(CgTest, Precisions);

    /** @brief 建立对称三对角矩阵。@tparam T 精度。@tparam I 索引。
     * @param n 阶数。@param diagonal 对角值。@pre n 正。
     * @return CSR 矩阵，对角为 2 时为一维 Dirichlet Poisson 矩阵。
     * @code auto a = tridiagonal<double>(20); @endcode */
    template <linalg::Scalar T, CsrIndex I = int>
    CsrMatrix<T, I> tridiagonal(int n, T diagonal = T(2)) {
        std::vector<I> offsets{0}, columns;
        std::vector<T> values;
        for (int row = 0; row < n; ++row) {
            if (row > 0) {
                columns.push_back(row - 1);
                values.push_back(-1);
            }
            columns.push_back(row);
            values.push_back(diagonal);
            if (row + 1 < n) {
                columns.push_back(row + 1);
                values.push_back(-1);
            }
            offsets.push_back(static_cast<I>(values.size()));
        }
        return CsrMatrix<T, I>(n, n, offsets, columns, values);
    }

    TYPED_TEST(CgTest, identitySolvesPoissonWithIndependentCpuResidual) {
        using T = TypeParam;
        constexpr int n = 31;
        CgOptions<T> options;
        options.relativeTolerance_ = std::same_as<T, float> ? T(2e-5) : T(1e-11);
        std::unique_ptr<Solver<T>> solver = std::make_unique<Cg<T>>(options, std::make_unique<Identity<T>>());
        solver->setup(tridiagonal<T>(n));
        const std::vector<T> ones(n, 1), zeros(n, 0);
        Vector<T> rhs{std::span<const T>(ones)}, x{std::span<const T>(zeros)};
        const auto report = solver->apply(rhs, x);
        ASSERT_EQ(report.status_, SolveStatus::Converged);
        EXPECT_GT(report.iterations_, 0);
        const auto solution = linalg::detail::hostCopy(x);
        double residualSquared = 0;
        for (int i = 0; i < n; ++i) {
            const double exact = double(i + 1) * double(n - i) / 2;
            EXPECT_NEAR(solution[i], exact, (std::same_as<T, float> ? 0.01 : 1e-8));
            const double residual = 1 -
                (2 * double(solution[i]) - (i ? double(solution[i - 1]) : 0) -
                 (i + 1 < n ? double(solution[i + 1]) : 0));
            residualSquared += residual * residual;
        }
        EXPECT_NEAR(report.residualNorm_, std::sqrt(residualSquared), (std::same_as<T, float> ? 1e-4 : 1e-12));
        EXPECT_LE(std::sqrt(residualSquared), double(options.relativeTolerance_) * std::sqrt(double(n)));
    }

    TYPED_TEST(CgTest, defaultIdentityPreservesPinnedStorageSupportsAliasAndReuse) {
        using T = TypeParam;
        Cg<T, std::int64_t> solver;
        auto matrix = tridiagonal<T, std::int64_t>(2);
        solver.setup(matrix);
        const std::vector<T> modified(4, 0);
        matrix.copyValuesFromHost(modified);
        const std::vector<T> b{1, 0}, initial{T(0.1), T(-0.2)};
        Vector<T> rhs(std::span<const T>(b), PinnedMemory{}), x(std::span<const T>(initial), PinnedMemory{});
        auto* address = x.data();
        ASSERT_EQ(solver.apply(rhs, x).status_, SolveStatus::Converged);
        EXPECT_EQ(address, x.data());
        EXPECT_EQ(x.memoryType(), MemoryType::Pinned);
        auto solution = linalg::detail::hostCopy(x);
        EXPECT_NEAR(solution[0], 2.0 / 3, 1e-5);
        EXPECT_NEAR(solution[1], 1.0 / 3, 1e-5);
        ASSERT_EQ(solver.apply(rhs, rhs).status_, SolveStatus::Converged);
        solution = linalg::detail::hostCopy(rhs);
        EXPECT_NEAR(solution[0], 2.0 / 3, 1e-5);
        EXPECT_NEAR(solution[1], 1.0 / 3, 1e-5);
        EXPECT_EQ(rhs.memoryType(), MemoryType::Pinned);
    }

    TYPED_TEST(CgTest, initialConvergenceZeroRhsAndIterationLimits) {
        using T = TypeParam;
        CgOptions<T> options;
        options.maxIterations_ = 0;
        Cg<T> initialOnly(options);
        initialOnly.setup(tridiagonal<T>(3));
        const std::vector<T> zeros(3, 0), b{1, 0, 0};
        Vector<T> rhs{std::span<const T>(zeros)}, x{std::span<const T>(zeros)};
        auto report = initialOnly.apply(rhs, x);
        EXPECT_EQ(report.status_, SolveStatus::Converged);
        EXPECT_EQ(report.iterations_, 0);
        EXPECT_EQ(report.relativeResidual_, 0);
        rhs.copyFromHost(b);
        report = initialOnly.apply(rhs, x);
        EXPECT_EQ(report.status_, SolveStatus::MaxIterations);
        EXPECT_EQ(report.iterations_, 0);
        EXPECT_NEAR(report.residualNorm_, 1, 1e-6);
        options.maxIterations_ = 1;
        Cg<T> bounded(options);
        bounded.setup(tridiagonal<T>(3));
        report = bounded.apply(rhs, x);
        EXPECT_EQ(report.status_, SolveStatus::MaxIterations);
        EXPECT_EQ(report.iterations_, 1);
        EXPECT_NEAR(report.residualNorm_, 0.5, 1e-6);
        // 对零右端项使用非零初值，验证绝对容差及相对残差定义。
        CgOptions<T> zeroOptions;
        zeroOptions.absoluteTolerance_ = T(1e-6);
        Cg<T> zeroSolver(zeroOptions);
        zeroSolver.setup(tridiagonal<T>(3));
        rhs.copyFromHost(zeros);
        report = zeroSolver.apply(rhs, x);
        EXPECT_EQ(report.status_, SolveStatus::Converged);
        EXPECT_EQ(report.relativeResidual_, report.residualNorm_);
    }

    TYPED_TEST(CgTest, refreshRestartsAndExactNonzeroInitialGuess) {
        using T = TypeParam;
        CgOptions<T> options;
        options.residualRefreshInterval_ = 1;
        options.maxIterations_ = 500;
        options.relativeTolerance_ = std::same_as<T, float> ? T(1e-5) : T(1e-10);
        Cg<T> solver(options);
        solver.setup(tridiagonal<T>(5, T(4)));
        const std::vector<T> b{3, 2, 2, 2, 3}, zero(5, 0), exact(5, 1);
        Vector<T> rhs{std::span<const T>(b)}, x{std::span<const T>(zero)};
        auto report = solver.apply(rhs, x);
        ASSERT_EQ(report.status_, SolveStatus::Converged);
        EXPECT_GT(report.iterations_, 1);
        for (auto value : linalg::detail::hostCopy(x)) {
            EXPECT_NEAR(value, 1, (std::same_as<T, float> ? 1e-4 : 1e-9));
        }
        x.copyFromHost(exact);
        report = solver.apply(rhs, x);
        EXPECT_EQ(report.status_, SolveStatus::Converged);
        EXPECT_EQ(report.iterations_, 0);
    }

    TYPED_TEST(CgTest, nonpositiveCurvatureAndNonfiniteRhsAreReported) {
        using T = TypeParam;
        const std::vector<T> b{1}, zero{0};
        Vector<T> rhs{std::span<const T>(b)}, x{std::span<const T>(zero)};
        Cg<T> solver;
        solver.setup(tridiagonal<T>(1, T(-1)));
        EXPECT_EQ(solver.apply(rhs, x).status_, SolveStatus::Breakdown);
        solver.setup(tridiagonal<T>(1, T(0)));
        EXPECT_EQ(solver.apply(rhs, x).status_, SolveStatus::Breakdown);
        solver.setup(tridiagonal<T>(1));
        const std::vector<T> invalid{std::numeric_limits<T>::quiet_NaN()};
        rhs.copyFromHost(invalid);
        EXPECT_EQ(solver.apply(rhs, x).status_, SolveStatus::NonFinite);
    }

    TYPED_TEST(CgTest, setupAndVectorContracts) {
        using T = TypeParam;
        Cg<T> solver;
        Vector<T> rhs(2), x(2), wrong(3);
        EXPECT_THROW(solver.apply(rhs, x), std::logic_error);
        solver.setup(tridiagonal<T>(2));
        EXPECT_THROW(solver.apply(wrong, x), std::invalid_argument);
        EXPECT_THROW(solver.apply(rhs, wrong), std::invalid_argument);
        const std::vector<int> offsets{0, 1, 2}, columns{0, 1};
        const std::vector<T> invalid{1, std::numeric_limits<T>::infinity()};
        EXPECT_THROW(solver.setup(CsrMatrix<T>(2, 2, offsets, columns, invalid)), std::invalid_argument);
        EXPECT_THROW(solver.apply(rhs, x), std::logic_error);
        EXPECT_NO_THROW(solver.setup(tridiagonal<T>(2)));
        EXPECT_THROW(solver.setup(CsrMatrix<T>{}), std::invalid_argument);
        EXPECT_THROW(solver.apply(rhs, x), std::logic_error);
    }

    TEST(CgOptionsTest, invalidOptionsRejectedWithoutCuda) {
        CgOptions<double> options;
        options.maxIterations_ = -1;
        EXPECT_THROW((Cg<double>(options)), std::invalid_argument);
        options = {};
        options.residualRefreshInterval_ = 0;
        EXPECT_THROW((Cg<double>(options)), std::invalid_argument);
        options = {};
        for (double value :
             {-1.0, 1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
            options.relativeTolerance_ = value;
            EXPECT_THROW((Cg<double>(options)), std::invalid_argument);
        }
        options = {};
        for (double value : {-1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
            options.absoluteTolerance_ = value;
            EXPECT_THROW((Cg<double>(options)), std::invalid_argument);
        }
    }
} // namespace

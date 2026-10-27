/** @file solver_test.cpp
 * @brief 验证平滑、预条件和求解接口的数学行为及生命周期。
 */
#include <Preconditioner/Amg.cuh>
#include <Preconditioner/Identity.cuh>
#include <Smoother/Jacobi.cuh>
#include <Solver/BiCgStab.cuh>
#include <gtest/gtest.h>
#include <limits>

/** @brief 包含求解器解析解与异常路径测试。 */
namespace {
    using namespace thermofem;
    /** @brief CUDA 求解测试环境。
     * @code SolverTest<double> fixture; @endcode
     */
    template <linalg::Scalar T>
    class SolverTest : public ::testing::Test {
    protected:
        /** @brief 激活设备，无参数、无额外约束。@return 无返回值。
         * @code SetUp(); @endcode
         */
        void SetUp() override {
            int count{};
            if (cudaGetDeviceCount(&count) != cudaSuccess || !count) {
                GTEST_SKIP() << "CUDA unavailable";
            }
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        }
    };
    using Precisions = ::testing::Types<float, double>;
    TYPED_TEST_SUITE(SolverTest, Precisions);
    /** @brief 建立一维 Poisson 或非对称三对角系统。@tparam T 精度。@tparam I 索引。
     * @param n 行数。@param nonsymmetric 是否非对称。
     * @pre n 正。@return 系数矩阵。@code auto a = poisson<double>(20); @endcode
     */
    template <linalg::Scalar T, CsrIndex I = int>
    CsrMatrix<T, I> poisson(int n, bool nonsymmetric = false) {
        std::vector<I> offsets{0}, columns;
        std::vector<T> values;
        for (int row = 0; row < n; ++row) {
            if (row) {
                columns.push_back(row - 1);
                values.push_back(-1);
            }
            columns.push_back(row);
            values.push_back(nonsymmetric ? T(3) : T(2));
            if (row + 1 < n) {
                columns.push_back(row + 1);
                values.push_back(nonsymmetric ? T(-0.5) : T(-1));
            }
            offsets.push_back(static_cast<I>(values.size()));
        }
        return CsrMatrix<T, I>(n, n, offsets, columns, values);
    }
    TYPED_TEST(SolverTest, jacobiMatchesAnalyticSweepsAndPreservesPolicy) {
        using T = TypeParam;
        const auto a = poisson<T>(2);
        std::unique_ptr<Smoother<T>> smoother = std::make_unique<Jacobi<T>>(2, T(1));
        smoother->setup(a);
        const std::vector<T> b{1, 0}, initial{0, 0};
        Vector<T> rhs(std::span<const T>(b), PinnedMemory{}), x(std::span<const T>(initial), PinnedMemory{});
        smoother->apply(rhs, x);
        auto values = linalg::detail::hostCopy(x);
        EXPECT_EQ(x.memoryType(), MemoryType::Pinned);
        EXPECT_NEAR(values[0], 0.5, 1e-6);
        EXPECT_NEAR(values[1], 0.25, 1e-6);
        // 别名时右端项必须在更新前保存。
        rhs.copyFromHost(b);
        smoother->apply(rhs, rhs);
        values = linalg::detail::hostCopy(rhs);
        EXPECT_NEAR(values[0], 0.75, 1e-6);
        EXPECT_NEAR(values[1], 0.25, 1e-6);
    }
    TYPED_TEST(SolverTest, jacobiSetupValidationAndReuse) {
        using T = TypeParam;
        Jacobi<T> smoother;
        Vector<T> rhs(2), x(2);
        EXPECT_THROW(smoother.apply(rhs, x), std::logic_error);
        EXPECT_THROW((Jacobi<T>(0)), std::invalid_argument);
        EXPECT_THROW((Jacobi<T>(1, std::numeric_limits<T>::quiet_NaN())), std::invalid_argument);
        smoother.setup(poisson<T>(2));
        const std::vector<int> offsets{0, 1, 2}, columns{1, 0};
        const std::vector<T> values{1, 1};
        const CsrMatrix<T> bad(2, 2, offsets, columns, values);
        EXPECT_THROW(smoother.setup(bad), std::invalid_argument);
        const std::vector<T> b{1, 1}, initial{0, 0};
        rhs.copyFromHost(b);
        x.copyFromHost(initial);
        EXPECT_NO_THROW(smoother.apply(rhs, x));
        Vector<T> wrong(3);
        EXPECT_THROW(smoother.apply(wrong, x), std::invalid_argument);
        smoother.setup(poisson<T>(3));
        EXPECT_THROW(smoother.apply(rhs, x), std::invalid_argument);
    }
    TYPED_TEST(SolverTest, contextOwnsSnapshotAndSupportsWideIndices) {
        using T = TypeParam;
        auto a = poisson<T, std::int64_t>(2);
        linalg::IterationContext<T, std::int64_t> context(a);
        const std::vector<T> zeros(4, 0), input{1, 2};
        a.copyValuesFromHost(zeros);
        const Vector<T> x{std::span<const T>(input)};
        Vector<T> y(2);
        context.multiply(x, y);
        const auto output = linalg::detail::hostCopy(y);
        EXPECT_NEAR(output[0], 0, 1e-6);
        EXPECT_NEAR(output[1], 3, 1e-6);
        EXPECT_THROW(context.multiply(x, const_cast<Vector<T>&>(x)), std::invalid_argument);
    }
    TYPED_TEST(SolverTest, amgCycleReducesErrorAndIsLinear) {
        using T = TypeParam;
        const auto a = poisson<T>(127);
        AmgOptions<T> amgOptions;
        amgOptions.coarseRows_ = 8;
        Amg<T> amg(amgOptions);
        Vector<T> uninitialized(127);
        EXPECT_THROW(amg.apply(uninitialized, uninitialized), std::logic_error);
        amg.setup(a);
        ASSERT_GE(amg.levelCount(), 3u);
        const std::vector<T> ones(127, 1), zeros(127, 0);
        Vector<T> rhs{std::span<const T>(ones)}, x{std::span<const T>(zeros)};
        linalg::IterationContext<T> context(a);
        Vector<T> residual(127);
        amg.apply(rhs, x);
        context.multiply(x, residual);
        context.axpy(T(-1), rhs, residual);
        const auto first = linalg::detail::hostCopy(x);
        // V-cycle 是误差校正，低频右端项的一次校正不保证残差二范数单调。
        T errorSquared{}, exactSquared{};
        for (std::size_t i = 0; i < first.size(); ++i) {
            const T exact = T(i + 1) * T(127 - i) / T(2);
            errorSquared += (first[i] - exact) * (first[i] - exact);
            exactSquared += exact * exact;
        }
        EXPECT_LT(std::sqrt(errorSquared / exactSquared), T(1));
        context.scale(T(2), rhs);
        amg.apply(rhs, x);
        const auto second = linalg::detail::hostCopy(x);
        const T tolerance = std::same_as<T, float> ? T(1e-2) : T(1e-9);
        for (std::size_t i = 0; i < first.size(); ++i) {
            EXPECT_NEAR(second[i], 2 * first[i], tolerance);
        }
        amg.apply(rhs, rhs);
        const auto aliased = linalg::detail::hostCopy(rhs);
        for (std::size_t i = 0; i < first.size(); ++i) {
            EXPECT_NEAR(aliased[i], second[i], tolerance);
        }
    }
    TYPED_TEST(SolverTest, amgCoarsePivotAndFailurePreservesState) {
        using T = TypeParam;
        const std::vector<int> offsets{0, 2, 4}, columns{0, 1, 0, 1};
        const std::vector<T> values{1, 2, 4, 3}, b{5, 10};
        const CsrMatrix<T> a(2, 2, offsets, columns, values);
        Amg<T> amg;
        amg.setup(a);
        Vector<T> rhs(std::span<const T>(b), PinnedMemory{}), x(2, PinnedMemory{});
        amg.apply(rhs, x);
        auto result = linalg::detail::hostCopy(x);
        EXPECT_NEAR(result[0], 1, 1e-5);
        EXPECT_NEAR(result[1], 2, 1e-5);
        const std::vector<int> fullOffsets{0, 2, 4}, fullColumns{0, 1, 0, 1};
        const std::vector<T> invalid{0, 1, 1, 1};
        EXPECT_THROW(amg.setup(CsrMatrix<T>(2, 2, fullOffsets, fullColumns, invalid)), std::invalid_argument);
        EXPECT_NO_THROW(amg.apply(rhs, x));
        EXPECT_EQ(x.memoryType(), MemoryType::Pinned);
        const std::vector<T> singular{1, 1, 1, 1};
        EXPECT_NO_THROW(amg.setup(CsrMatrix<T>(2, 2, fullOffsets, fullColumns, singular)));
        EXPECT_EQ(amg.coarseSolver(), "Jacobi (singular LU)");
        EXPECT_NO_THROW(amg.apply(rhs, x));
        AmgOptions<T> options;
        options.maxLevels_ = 1;
        options.coarseRows_ = 2;
        Amg<T> bounded(options);
        EXPECT_NO_THROW(bounded.setup(poisson<T>(100)));
        EXPECT_EQ(bounded.levelCount(), 1u);
        EXPECT_EQ(bounded.coarseSolver(), "Jacobi (coarsening stopped)");
        options.maxLevels_ = 0;
        EXPECT_THROW((Amg<T>(options)), std::invalid_argument);
    }
    TYPED_TEST(SolverTest, bicgstabAmgSolvesPoissonAndReportsTrueResidual) {
        using T = TypeParam;
        const int n = 63;
        const auto a = poisson<T>(n);
        BiCgStabOptions<T> options;
        options.relativeTolerance_ = std::same_as<T, float> ? T(2e-4) : T(1e-9);
        BiCgStab<T> solver(options);
        solver.setup(a);
        const std::vector<T> b(n, 1), initial(n, 0);
        Vector<T> rhs(std::span<const T>(b), PinnedMemory{}), x(std::span<const T>(initial), PinnedMemory{});
        auto* address = x.data();
        const auto result = solver.apply(rhs, x);
        ASSERT_EQ(result.status_, SolveStatus::Converged);
        EXPECT_LT(result.iterations_, 100);
        EXPECT_EQ(x.data(), address);
        EXPECT_EQ(x.memoryType(), MemoryType::Pinned);
        const auto actual = linalg::detail::hostCopy(x);
        double residualSquared{};
        for (int i = 0; i < n; ++i) {
            const double exact = double(i + 1) * double(n - i) / 2;
            EXPECT_NEAR(actual[i], exact, exact * (std::same_as<T, float> ? 1e-3 : 1e-8));
            const double residual =
                1 - 2 * double(actual[i]) + (i ? double(actual[i - 1]) : 0) + (i + 1 < n ? double(actual[i + 1]) : 0);
            residualSquared += residual * residual;
        }
        EXPECT_NEAR(result.residualNorm_, std::sqrt(residualSquared), (std::same_as<T, float> ? 1e-3 : 1e-10));
        EXPECT_LE(result.relativeResidual_, options.relativeTolerance_);
        const auto repeated = solver.apply(rhs, x);
        EXPECT_EQ(repeated.status_, SolveStatus::Converged);
        EXPECT_EQ(repeated.iterations_, 0);
        // 输入输出别名时初值等于右端项，仍应求得同一解。
        rhs.copyFromHost(b);
        EXPECT_EQ(solver.apply(rhs, rhs).status_, SolveStatus::Converged);
    }
    TYPED_TEST(SolverTest, bicgstabIdentitySolvesNonsymmetricWithResidualRefresh) {
        using T = TypeParam;
        const int n = 80;
        const auto a = poisson<T>(n, true);
        std::vector<T> exact(n), b(n), initial(n, T(0.2));
        for (int i = 0; i < n; ++i) {
            exact[i] = T(1) + T(i % 7) / T(7);
        }
        for (int i = 0; i < n; ++i) {
            b[i] = 3 * exact[i] - (i ? exact[i - 1] : 0) - T(0.5) * (i + 1 < n ? exact[i + 1] : 0);
        }
        BiCgStabOptions<T> options;
        options.residualRefreshInterval_ = 3;
        options.relativeTolerance_ = std::same_as<T, float> ? T(1e-5) : T(1e-11);
        std::unique_ptr<Solver<T>> solver = std::make_unique<BiCgStab<T>>(options, std::make_unique<Identity<T>>());
        solver->setup(a);
        Vector<T> rhs{std::span<const T>(b)}, x{std::span<const T>(initial)};
        const auto report = solver->apply(rhs, x);
        ASSERT_EQ(report.status_, SolveStatus::Converged);
        const auto actual = linalg::detail::hostCopy(x);
        for (int i = 0; i < n; ++i) {
            EXPECT_NEAR(actual[i], exact[i], (std::same_as<T, float> ? 1e-4 : 1e-9));
        }
    }
    TYPED_TEST(SolverTest, bicgstabZeroRhsAndInitialGuess) {
        using T = TypeParam;
        BiCgStabOptions<T> options;
        options.absoluteTolerance_ = std::same_as<T, float> ? T(1e-5) : T(1e-12);
        BiCgStab<T> solver(options, std::make_unique<Identity<T>>());
        solver.setup(poisson<T>(5));
        const std::vector<T> zeros(5, 0), ones(5, 1);
        Vector<T> rhs{std::span<const T>(zeros)}, x{std::span<const T>(zeros)};
        auto report = solver.apply(rhs, x);
        EXPECT_EQ(report.status_, SolveStatus::Converged);
        EXPECT_EQ(report.iterations_, 0);
        x.copyFromHost(ones);
        report = solver.apply(rhs, x);
        EXPECT_EQ(report.status_, SolveStatus::Converged);
        EXPECT_LE(report.residualNorm_, options.absoluteTolerance_);
        EXPECT_EQ(report.relativeResidual_, report.residualNorm_);
    }
    TYPED_TEST(SolverTest, bicgstabReportsLimitBreakdownAndNonfinite) {
        using T = TypeParam;
        const auto a = poisson<T>(3, true);
        const std::vector<T> b{1, 2, 3}, zeros(3, 0);
        Vector<T> rhs{std::span<const T>(b)}, x{std::span<const T>(zeros)};
        BiCgStabOptions<T> options;
        options.maxIterations_ = 0;
        BiCgStab<T> limited(options, std::make_unique<Identity<T>>());
        limited.setup(a);
        auto result = limited.apply(rhs, x);
        EXPECT_EQ(result.status_, SolveStatus::MaxIterations);
        EXPECT_EQ(result.iterations_, 0);
        options.maxIterations_ = 1;
        options.relativeTolerance_ = 0;
        BiCgStab<T> oneStep(options, std::make_unique<Identity<T>>());
        oneStep.setup(a);
        result = oneStep.apply(rhs, x);
        EXPECT_EQ(result.status_, SolveStatus::MaxIterations);
        EXPECT_EQ(result.iterations_, 1);
        const std::vector<int> offsets{0, 0, 0, 0}, columns;
        const std::vector<T> values;
        const CsrMatrix<T> zeroMatrix(3, 3, offsets, columns, values);
        BiCgStab<T> solver({}, std::make_unique<Identity<T>>());
        solver.setup(zeroMatrix);
        x.copyFromHost(zeros);
        result = solver.apply(rhs, x);
        EXPECT_EQ(result.status_, SolveStatus::Breakdown);
        EXPECT_NEAR(result.residualNorm_, std::sqrt(14.0), 1e-5);
        auto invalid = b;
        invalid[1] = std::numeric_limits<T>::quiet_NaN();
        rhs.copyFromHost(invalid);
        EXPECT_EQ(solver.apply(rhs, x).status_, SolveStatus::NonFinite);
    }
    TYPED_TEST(SolverTest, bicgstabLifecycleWideIndicesAndFailedSetup) {
        using T = TypeParam;
        BiCgStab<T, std::int64_t> solver;
        const std::vector<T> b{1, 0}, initial{0, 0};
        Vector<T> rhs{std::span<const T>(b)}, x{std::span<const T>(initial)};
        EXPECT_THROW(solver.apply(rhs, x), std::logic_error);
        solver.setup(poisson<T, std::int64_t>(2));
        EXPECT_EQ(solver.apply(rhs, x).status_, SolveStatus::Converged);
        Vector<T> wrong(3);
        EXPECT_THROW(solver.apply(wrong, x), std::invalid_argument);
        EXPECT_THROW(solver.setup(CsrMatrix<T, std::int64_t>{}), std::invalid_argument);
        EXPECT_THROW(solver.apply(rhs, x), std::logic_error);
        solver.setup(poisson<T, std::int64_t>(2));
        EXPECT_EQ(solver.apply(rhs, x).status_, SolveStatus::Converged);
        BiCgStabOptions<T> options;
        options.relativeTolerance_ = -1;
        EXPECT_THROW((BiCgStab<T>(options)), std::invalid_argument);
    }
    TYPED_TEST(SolverTest, amgAcceptsIndependentSmootherFactory) {
        using T = TypeParam;
        int created{};
        Amg<T> amg({}, [&created]() -> std::unique_ptr<Smoother<T>> {
            ++created;
            return std::make_unique<Jacobi<T>>(3, T(0.6));
        });
        amg.setup(poisson<T>(255));
        EXPECT_EQ(static_cast<std::size_t>(created), 2 * (amg.levelCount() - 1));
        Amg<T> invalid({}, []() -> std::unique_ptr<Smoother<T>> { return {}; });
        EXPECT_THROW(invalid.setup(poisson<T>(255)), std::invalid_argument);
    }
    TYPED_TEST(SolverTest, solverPreservesAllSupportedMemoryPolicies) {
        using T = TypeParam;
        BiCgStab<T> solver;
        solver.setup(poisson<T>(3));
        std::vector<MemoryStrategy> policies{GlobalMemory{}, PinnedMemory{}};
        int supported{};
        if (cudaDeviceGetAttribute(&supported, cudaDevAttrCanMapHostMemory, 0) == cudaSuccess && supported) {
            policies.emplace_back(MappedMemory{});
        }
        if (cudaDeviceGetAttribute(&supported, cudaDevAttrManagedMemory, 0) == cudaSuccess && supported) {
            policies.emplace_back(ManagedMemory{});
        }
        const std::vector<T> b{1, 0, 1}, initial(3, 0);
        for (const auto& inputPolicy : policies) {
            const Vector<T> rhs(std::span<const T>(b), inputPolicy);
            for (const auto& outputPolicy : policies) {
                Vector<T> x(std::span<const T>(initial), outputPolicy);
                const auto type = x.memoryType();
                auto* address = x.data();
                ASSERT_EQ(solver.apply(rhs, x).status_, SolveStatus::Converged);
                EXPECT_EQ(x.memoryType(), type);
                EXPECT_EQ(x.data(), address);
                for (T value : linalg::detail::hostCopy(x)) {
                    EXPECT_NEAR(value, 1, 1e-5);
                }
            }
        }
    }
    TYPED_TEST(SolverTest, emptyPatternKeepsOwnerDeviceAndRestoresCaller) {
        using T = TypeParam;
        int count{};
        ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
        if (count < 2) {
            GTEST_SKIP() << "Requires two devices";
        }
        ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
        const std::vector<int> offsets{0, 0, 0}, columns;
        const std::vector<T> values, ones(2, 1), zeros(2, 0);
        const CsrMatrix<T> a(2, 2, offsets, columns, values);
        Vector<T> rhs{std::span<const T>(ones)}, x{std::span<const T>(zeros)};
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        BiCgStab<T> solver({}, std::make_unique<Identity<T>>());
        solver.setup(a);
        EXPECT_EQ(solver.apply(rhs, x).status_, SolveStatus::Breakdown);
        int current{};
        ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
        EXPECT_EQ(current, 0);
        Vector<T> wrong(2);
        EXPECT_THROW(solver.apply(wrong, x), std::invalid_argument);
        ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
        EXPECT_EQ(current, 0);
    }
} // namespace

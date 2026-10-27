/** @file multigrid_operations_test.cpp
 * @brief 独立验证求解流程依赖的对角线、残差、延拓、Galerkin 与粗层 LU 运算。
 */
#include <LinearAlgebra/CoarseLu.cuh>
#include <LinearAlgebra/Diagonal.cuh>
#include <LinearAlgebra/IterationContext.cuh>
#include <Multigrid/GpuAggregation.cuh>
#include <gtest/gtest.h>

/** @brief 独立数学运算测试，不依赖任何求解组件。 */
namespace {
    using namespace thermofem;
    /** @brief 设备测试环境。@code MultigridOperationTest<double> fixture; @endcode */
    template <linalg::Scalar T>
    class MultigridOperationTest : public ::testing::Test {
    protected:
        /** @brief 激活 CUDA 设备，无参数、无额外约束。@note 无返回值。@code SetUp(); @endcode */
        void SetUp() override {
            int count{};
            if (cudaGetDeviceCount(&count) != cudaSuccess || !count) {
                GTEST_SKIP() << "CUDA unavailable";
            }
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        }
    };
    using Precisions = ::testing::Types<float, double>;
    TYPED_TEST_SUITE(MultigridOperationTest, Precisions);
    TYPED_TEST(MultigridOperationTest, diagonalResidualAndGalerkinMatchHandCalculation) {
        using T = TypeParam;
        const std::vector<int> offsets{0, 2, 5, 8, 10}, columns{0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
        const std::vector<T> values{2, -1, -1, 2, -1, -1, 2, -1, -1, 2};
        const CsrMatrix<T> a(4, 4, offsets, columns, values);
        const auto inverse = linalg::detail::hostCopy(linalg::inverseDiagonal(a));
        for (auto value : inverse) {
            EXPECT_EQ(value, T(0.5));
        }
        multigrid::GpuAggregation setup;
        const auto native = multigrid::toNativeMatrix(a);
        const auto graph = setup.strength(native, .25);
        const auto transfer = setup.aggregate(graph, 8, true);
        const auto coarse = setup.galerkin(native, transfer);
        // 两阶段匹配把四点聚合为一组，P^T*A*P=2，不使用平滑延拓。
        ASSERT_EQ(coarse.rowCount(), 1u);
        EXPECT_EQ(linalg::detail::hostCopy(coarse.values()), (std::vector<double>{2}));
        const std::vector<T> ones(4, 1), zeros(4, 0);
        Vector<T> x{std::span<const T>(ones)}, rhs{std::span<const T>(zeros)}, residual(4);
        linalg::IterationContext<T> context(a);
        context.residual(rhs, x, residual);
        const auto actual = linalg::detail::hostCopy(residual);
        EXPECT_EQ(actual, (std::vector<T>{-1, 0, 0, -1}));
        // 单独验证 Jacobi 更新原语，不通过平滑器组件调用。
        Vector<T> diagonal{std::span<const T>(inverse)};
        context.clear(x);
        context.clear(residual);
        rhs.copyFromHost(ones);
        context.jacobi(T(1), diagonal, rhs, residual, x);
        for (T value : linalg::detail::hostCopy(x)) {
            EXPECT_EQ(value, T(0.5));
        }
        EXPECT_THROW(context.residual(rhs, x, rhs), std::invalid_argument);
        EXPECT_THROW(setup.strength(native, -1), std::invalid_argument);
    }
    TYPED_TEST(MultigridOperationTest, coarseLuPivotAndDuplicateDiagonal) {
        using T = TypeParam;
        const std::vector<int> offsets{0, 2, 4}, columns{0, 0, 0, 1};
        const std::vector<T> values{1, 1, 4, 3}, b{2, 7};
        const CsrMatrix<T> a(2, 2, offsets, columns, values);
        const auto diagonal = linalg::detail::hostCopy(linalg::inverseDiagonal(a));
        EXPECT_NEAR(diagonal[0], 0.5, 1e-6);
        EXPECT_NEAR(diagonal[1], 1.0 / 3, 1e-6);
        linalg::CoarseLu<T> lu(a);
        Vector<T> rhs{std::span<const T>(b)}, x(2);
        lu.apply(rhs, x);
        const auto result = linalg::detail::hostCopy(x);
        EXPECT_NEAR(result[0], 1, 1e-6);
        EXPECT_NEAR(result[1], 1, 1e-6);
        Vector<T> wrong(3);
        EXPECT_THROW(lu.apply(wrong, x), std::invalid_argument);
        const std::vector<T> invalid{0, 0, 4, 3};
        const CsrMatrix<T> bad(2, 2, offsets, columns, invalid);
        EXPECT_THROW(linalg::inverseDiagonal(bad), std::invalid_argument);
        EXPECT_THROW((linalg::CoarseLu<T>(bad)), std::invalid_argument);
    }
    TYPED_TEST(MultigridOperationTest, prolongationKeepsOwnerAndRestoresCallerDevice) {
        using T = TypeParam;
        int count{};
        ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
        if (count < 2) {
            GTEST_SKIP() << "Requires two devices";
        }
        ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
        const std::vector<int> offsets{0, 2, 4}, columns{0, 1, 0, 1};
        const std::vector<T> values{2, -1, -1, 2}, ones(2, 1);
        const CsrMatrix<T> a(2, 2, offsets, columns, values);
        Vector<T> rhs{std::span<const T>(ones)};
        const auto native = multigrid::toNativeMatrix(a);
        multigrid::GpuAggregation setup;
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        const auto p = setup.strength(native, .25);
        EXPECT_EQ(p.rowOffsets().deviceId(), 1);
        int current{};
        ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
        EXPECT_EQ(current, 0);
        linalg::CoarseLu<T> lu(a);
        Vector<T> wrong(2);
        EXPECT_THROW(lu.apply(rhs, wrong), std::invalid_argument);
    }
} // namespace

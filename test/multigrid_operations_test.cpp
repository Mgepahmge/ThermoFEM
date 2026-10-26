/** @file multigrid_operations_test.cpp
 * @brief 独立验证求解流程依赖的对角线、残差、延拓、Galerkin 与粗层 LU 运算。
 */
#include <LinearAlgebra/Diagonal.cuh>
#include <LinearAlgebra/CoarseLu.cuh>
#include <LinearAlgebra/IterationContext.cuh>
#include <Multigrid/Aggregation.cuh>
#include <gtest/gtest.h>

/** @brief 独立数学运算测试，不依赖任何求解组件。 */
namespace {
using namespace thermofem;
/** @brief 设备测试环境。@code MultigridOperationTest<double> fixture; @endcode */
template<linalg::Scalar T> class MultigridOperationTest : public ::testing::Test {
protected:
    /** @brief 激活 CUDA 设备，无参数、无额外约束。@return 无返回值。@code SetUp(); @endcode */
    void SetUp() override {
        int count{};
        if (cudaGetDeviceCount(&count) != cudaSuccess || !count) { GTEST_SKIP() << "CUDA unavailable"; }
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
    for (auto value : inverse) { EXPECT_EQ(value, T(0.5)); }
    const auto p = multigrid::prolongation(a, T(2)/3, T(0.25));
    ASSERT_EQ(p.columnCount(), 2u);
    const auto r = linalg::transpose(p);
    const auto coarse = multigrid::galerkin(a, p, r);
    const auto coarseValues = linalg::detail::hostCopy(coarse.values());
    ASSERT_EQ(coarseValues.size(), 4u);
    EXPECT_NEAR(coarseValues[0], 2.0/3, 1e-6);
    EXPECT_NEAR(coarseValues[1], -1.0/9, 1e-6);
    EXPECT_NEAR(coarseValues[2], -1.0/9, 1e-6);
    EXPECT_NEAR(coarseValues[3], 2.0/3, 1e-6);
    const std::vector<T> ones(4, 1), zeros(4, 0);
    Vector<T> x{std::span<const T>(ones)}, rhs{std::span<const T>(zeros)}, residual(4);
    linalg::IterationContext<T> context(a);
    context.residual(rhs, x, residual);
    const auto actual = linalg::detail::hostCopy(residual);
    EXPECT_EQ(actual, (std::vector<T>{-1, 0, 0, -1}));
    EXPECT_THROW(context.residual(rhs, x, rhs), std::invalid_argument);
    EXPECT_THROW(multigrid::galerkin(a, r, p), std::invalid_argument);
    EXPECT_THROW(multigrid::prolongation(a, T(1), T(-1)), std::invalid_argument);
}
TYPED_TEST(MultigridOperationTest, coarseLuPivotAndDuplicateDiagonal) {
    using T = TypeParam;
    const std::vector<int> offsets{0, 2, 4}, columns{0, 0, 0, 1};
    const std::vector<T> values{1, 1, 4, 3}, b{2, 7};
    const CsrMatrix<T> a(2, 2, offsets, columns, values);
    const auto diagonal = linalg::detail::hostCopy(linalg::inverseDiagonal(a));
    EXPECT_NEAR(diagonal[0], 0.5, 1e-6); EXPECT_NEAR(diagonal[1], 1.0/3, 1e-6);
    linalg::CoarseLu<T> lu(a);
    Vector<T> rhs{std::span<const T>(b)}, x(2);
    lu.apply(rhs, x);
    const auto result = linalg::detail::hostCopy(x);
    EXPECT_NEAR(result[0], 1, 1e-6); EXPECT_NEAR(result[1], 1, 1e-6);
    Vector<T> wrong(3);
    EXPECT_THROW(lu.apply(wrong, x), std::invalid_argument);
    const std::vector<T> invalid{0, 0, 4, 3};
    const CsrMatrix<T> bad(2, 2, offsets, columns, invalid);
    EXPECT_THROW(linalg::inverseDiagonal(bad), std::invalid_argument);
    EXPECT_THROW((linalg::CoarseLu<T>(bad)), std::invalid_argument);
}
} // namespace

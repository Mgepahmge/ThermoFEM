/** @file solver_test.cpp
 * @brief 验证平滑、预条件和求解接口的数学行为及生命周期。
 */
#include <Smoother/Jacobi.cuh>
#include <gtest/gtest.h>
#include <limits>

/** @brief 包含求解器解析解与异常路径测试。 */
namespace {
using namespace thermofem;
/** @brief CUDA 求解测试环境。
 * @code SolverTest<double> fixture; @endcode
 */
template<linalg::Scalar T> class SolverTest : public ::testing::Test {
protected:
    /** @brief 激活设备，无参数、无额外约束。@return 无返回值。
     * @code SetUp(); @endcode
     */
    void SetUp() override {
        int count{};
        if (cudaGetDeviceCount(&count) != cudaSuccess || !count) { GTEST_SKIP() << "CUDA unavailable"; }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }
};
using Precisions = ::testing::Types<float, double>;
TYPED_TEST_SUITE(SolverTest, Precisions);
/** @brief 建立一维 Poisson 或非对称三对角系统。@tparam T 精度。@tparam I 索引。
 * @param n 行数。@param nonsymmetric 是否非对称。
 * @pre n 正。@return 系数矩阵。@code auto a = poisson<double>(20); @endcode
 */
template<linalg::Scalar T, CsrIndex I = int>
CsrMatrix<T, I> poisson(int n, bool nonsymmetric = false) {
    std::vector<I> offsets{0}, columns;
    std::vector<T> values;
    for (int row = 0; row < n; ++row) {
        if (row) { columns.push_back(row-1); values.push_back(-1); }
        columns.push_back(row); values.push_back(nonsymmetric ? T(3) : T(2));
        if (row+1 < n) { columns.push_back(row+1); values.push_back(nonsymmetric ? T(-0.5) : T(-1)); }
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
    EXPECT_NEAR(values[0], 0.5, 1e-6); EXPECT_NEAR(values[1], 0.25, 1e-6);
    // 别名时右端项必须在更新前保存。
    rhs.copyFromHost(b); smoother->apply(rhs, rhs);
    values = linalg::detail::hostCopy(rhs);
    EXPECT_NEAR(values[0], 0.75, 1e-6); EXPECT_NEAR(values[1], 0.25, 1e-6);
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
    rhs.copyFromHost(b); x.copyFromHost(initial);
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
    EXPECT_NEAR(output[0], 0, 1e-6); EXPECT_NEAR(output[1], 3, 1e-6);
    EXPECT_THROW(context.multiply(x, const_cast<Vector<T>&>(x)), std::invalid_argument);
}
} // namespace

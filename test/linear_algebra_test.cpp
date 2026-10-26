/** @file linear_algebra_test.cpp
 * @brief 对照解析结果验证 cuBLAS、cuSPARSE 封装及各类运算符。
 */
#include <LinearAlgebra/LinearAlgebra.cuh>
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <vector>

/** @brief 包含浮点精度、内存类型及索引组合的线性代数测试。 */
namespace {
using namespace thermofem;
namespace la = thermofem::linalg;

/** @brief 建立真实 CUDA 测试环境，设备不可用时明确跳过。
 * @tparam T 浮点精度。@code LinearAlgebraTest<double> fixture; @endcode
 */
template<la::Scalar T> class LinearAlgebraTest : public ::testing::Test {
protected:
    /** @brief 选择首个设备，无参数。@pre 无额外约束。@return 无返回值。
     * @code SetUp(); @endcode
     */
    void SetUp() override {
        int count{};
        const auto status = cudaGetDeviceCount(&count);
        if (status != cudaSuccess || count == 0) {
            GTEST_SKIP() << (status == cudaSuccess ? "No CUDA device" : cudaGetErrorString(status));
        }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }
};
using Precisions = ::testing::Types<float, double>;
TYPED_TEST_SUITE(LinearAlgebraTest, Precisions);

/** @brief 获取当前设备支持的内存策略，无参数。@pre CUDA 设备可用。@return 策略列表。
 * @code auto policies = supportedStrategies(); @endcode
 */
std::vector<MemoryStrategy> supportedStrategies() {
    std::vector<MemoryStrategy> result{GlobalMemory{}, PinnedMemory{}};
    int supported{};
    if (cudaDeviceGetAttribute(&supported, cudaDevAttrCanMapHostMemory, 0) == cudaSuccess && supported) { result.emplace_back(MappedMemory{}); }
    if (cudaDeviceGetAttribute(&supported, cudaDevAttrManagedMemory, 0) == cudaSuccess && supported) { result.emplace_back(ManagedMemory{}); }
    return result;
}
/** @brief 检查缓冲区与主机期望值一致。@tparam T 精度。@param buffer 结果。@param expected 期望值。
 * @pre 尺寸相同且数据已同步。@return 无返回值。@code expectValues(result, {1, 2}); @endcode
 */
template<la::Scalar T> void expectValues(const CudaBuffer<T>& buffer, std::initializer_list<T> expected) {
    std::vector<T> actual(buffer.size());
    buffer.copyToHost(actual);
    ASSERT_EQ(actual.size(), expected.size());
    std::size_t index{};
    const double tolerance = std::same_as<T, float> ? 1e-5 : 1e-12;
    for (const auto value : expected) { EXPECT_NEAR(actual[index++], value, tolerance); }
}
/** @brief 验证 CSR 的数学矩阵值而非存储顺序。@tparam T 精度。@tparam I 索引。
 * @param matrix 结果。@param expected 行主序期望值。@pre 尺寸匹配。@return 无返回值。
 * @code expectSparse(matrix, {2, 0, 0, 3}); @endcode
 */
template<la::Scalar T, CsrIndex I>
void expectSparse(const CsrMatrix<T, I>& matrix, std::initializer_list<T> expected) {
    auto offsets = la::detail::hostCopy(matrix.rowOffsets());
    auto columns = la::detail::hostCopy(matrix.columnIndices());
    auto entries = la::detail::hostCopy(matrix.values());
    std::vector<T> actual(matrix.rowCount() * matrix.columnCount(), T{});
    for (std::size_t row = 0; row < matrix.rowCount(); ++row) {
        for (auto index = offsets[row]; index < offsets[row + 1]; ++index) {
            const auto position = static_cast<std::size_t>(index);
            actual[row * matrix.columnCount() + static_cast<std::size_t>(columns[position])] += entries[position];
        }
    }
    ASSERT_EQ(actual.size(), expected.size());
    std::size_t index{};
    const double tolerance = std::same_as<T, float> ? 1e-5 : 1e-12;
    for (const auto value : expected) { EXPECT_NEAR(actual[index++], value, tolerance); }
}

/** @brief 验证全部内存类型组合下的向量封装和运算符，无参数。
 * @pre CUDA 可用。@return 无返回值。@code ./build/test/thermofem_tests --gtest_filter=*vectorArithmetic* @endcode
 */
TYPED_TEST(LinearAlgebraTest, vectorArithmetic) {
    using T = TypeParam;
    for (const auto& leftStrategy : supportedStrategies()) {
        for (const auto& rightStrategy : supportedStrategies()) {
            const std::array<T, 3> leftValues{1, -2, 3}, rightValues{4, 5, -6};
            Vector<T> x(leftValues, leftStrategy), y(rightValues, rightStrategy);
            auto sum = x + y;
            EXPECT_EQ(sum.memoryStrategy().index(), leftStrategy.index());
            expectValues<T>(sum, {5, 3, -3});
            expectValues<T>(x - y, {-3, -7, 9});
            expectValues<T>(-x, {-1, 2, -3});
            expectValues<T>(+x, {1, -2, 3});
            expectValues<T>(2 * x, {2, -4, 6});
            expectValues<T>(x * 2, {2, -4, 6});
            expectValues<T>(x / 2, {T{0.5}, -1, T{1.5}});
            EXPECT_NEAR(x * y, -24, 1e-5);
            EXPECT_NEAR(la::dot(x, y), -24, 1e-5);
            EXPECT_NEAR(la::norm1(x), 6, 1e-5);
            EXPECT_NEAR(la::norm2(x), std::sqrt(14.0), 1e-5);
            expectValues<T>(la::axpy(T{2}, x, y), {6, 1, 0});
            auto copy = x;
            copy += copy;
            copy -= x;
            copy *= 2;
            copy /= 2;
            expectValues<T>(copy, {1, -2, 3});
            EXPECT_EQ(copy.memoryStrategy().index(), leftStrategy.index());
            EXPECT_THROW(x / 0, std::domain_error);
            expectValues<T>(x, {1, -2, 3});
        }
    }
}

/** @brief 验证非方阵行主序、GEMV、GEMM、GEAM 和矩阵运算符，无参数。
 * @pre CUDA 可用。@return 无返回值。@code ./build/test/thermofem_tests --gtest_filter=*denseArithmetic* @endcode
 */
TYPED_TEST(LinearAlgebraTest, denseArithmetic) {
    using T = TypeParam;
    for (const auto& strategy : supportedStrategies()) {
        const std::array<T, 6> aValues{1, 2, 3, 4, 5, 6};
        const std::array<T, 6> bValues{7, 8, 9, 10, 11, 12};
        const std::array<T, 3> xValues{1, 2, 3};
        DenseMatrix<T> a(2, 3, strategy), b(3, 2, PinnedMemory{});
        a.copyFromHost(aValues); b.copyFromHost(bValues);
        Vector<T> x(xValues, PinnedMemory{});
        expectValues<T>(a * x, {14, 32});
        auto product = a * b;
        EXPECT_EQ(product.rowCount(), 2);
        EXPECT_EQ(product.columnCount(), 2);
        EXPECT_EQ(product.values().memoryStrategy().index(), strategy.index());
        expectValues<T>(product.values(), {58, 64, 139, 154});
        auto transposed = la::transpose(a);
        EXPECT_EQ(transposed.rowCount(), 3);
        EXPECT_EQ(transposed.columnCount(), 2);
        expectValues<T>(transposed.values(), {1, 4, 2, 5, 3, 6});
        expectValues<T>((a + a).values(), {2, 4, 6, 8, 10, 12});
        expectValues<T>((a - a).values(), {0, 0, 0, 0, 0, 0});
        expectValues<T>((-a).values(), {-1, -2, -3, -4, -5, -6});
        auto copy = a;
        copy += a; copy -= a; copy *= 2; copy /= 2;
        expectValues<T>(copy.values(), {1, 2, 3, 4, 5, 6});
        copy *= b;
        expectValues<T>(copy.values(), {58, 64, 139, 154});
        EXPECT_NEAR(la::frobeniusNorm(a), std::sqrt(91.0), 1e-5);
        EXPECT_THROW(a / 0, std::domain_error);
    }
}

/** @brief 验证两种索引类型的稀疏运算和重复列输入。
 * @tparam T 精度。@tparam I 索引。@param strategy 内存策略。
 * @pre CUDA 可用。@return 无返回值。@code checkSparse<double, int>(GlobalMemory{}); @endcode
 */
template<la::Scalar T, CsrIndex I> void checkSparse(MemoryStrategy strategy) {
    const std::array<I, 3> offsets{0, 3, 4};
    const std::array<I, 4> columns{2, 0, 2, 1};
    const std::array<T, 4> entries{1, 2, 3, 5};
    CsrMatrix<T, I> a(2, 3, offsets, columns, entries, strategy);
    const std::array<I, 4> bOffsets{0, 1, 2, 4};
    const std::array<I, 4> bColumns{0, 1, 1, 0};
    const std::array<T, 4> bEntries{1, 2, 4, 3};
    CsrMatrix<T, I> b(3, 2, bOffsets, bColumns, bEntries, PinnedMemory{});
    const std::array<T, 3> xValues{1, 2, 3};
    Vector<T> x(xValues, PinnedMemory{});
    expectValues<T>(a * x, {14, 10});
    DenseMatrix<T> dense(3, 2, PinnedMemory{});
    dense.copyFromHost(std::array<T, 6>{1, 0, 0, 2, 3, 4});
    expectValues<T>((a * dense).values(), {14, 16, 0, 10});
    auto product = a * b;
    expectSparse<T, I>(product, {14, 16, 0, 10});
    EXPECT_EQ(product.values().memoryStrategy().index(), strategy.index());
    expectSparse<T, I>(a + a, {4, 0, 8, 0, 10, 0});
    expectSparse<T, I>(a - a, {0, 0, 0, 0, 0, 0});
    auto transposed = la::transpose(a);
    expectSparse<T, I>(transposed, {2, 0, 0, 5, 4, 0});
    expectSparse<T, I>(-a, {-2, 0, -4, 0, -5, 0});
    expectSparse<T, I>(a * 2, {4, 0, 8, 0, 10, 0});
    EXPECT_NEAR(la::frobeniusNorm(a), std::sqrt(45.0), 1e-5);
    auto copy = a;
    copy += copy; copy -= a; copy *= 2; copy /= 2;
    expectSparse<T, I>(copy, {2, 0, 4, 0, 5, 0});
    copy *= b;
    expectSparse<T, I>(copy, {14, 16, 0, 10});
    EXPECT_THROW(a / 0, std::domain_error);
}
/** @brief 验证全部稀疏策略及索引组合，无参数。@pre CUDA 可用。@return 无返回值。
 * @code ./build/test/thermofem_tests --gtest_filter=*sparseArithmetic* @endcode
 */
TYPED_TEST(LinearAlgebraTest, sparseArithmetic) {
    for (const auto& strategy : supportedStrategies()) {
        checkSparse<TypeParam, std::int32_t>(strategy);
        checkSparse<TypeParam, std::int64_t>(strategy);
    }
}

/** @brief 验证零内维度、零行列、零非零元等边界，无参数。@pre CUDA 可用。@return 无返回值。
 * @code ./build/test/thermofem_tests --gtest_filter=*zeroDimensions* @endcode
 */
TYPED_TEST(LinearAlgebraTest, zeroDimensions) {
    using T = TypeParam;
    DenseMatrix<T> a(2, 0, PinnedMemory{}), b(0, 3);
    expectValues<T>((a * b).values(), {0, 0, 0, 0, 0, 0});
    Vector<T> empty;
    expectValues<T>(a * empty, {0, 0});
    EXPECT_EQ(la::dot(empty, empty), 0);
    EXPECT_EQ(la::norm1(empty), 0);
    EXPECT_EQ(la::norm2(empty), 0);
    EXPECT_EQ(la::transpose(a).rowCount(), 0);
    const std::array<int, 3> offsets{0, 0, 0};
    CsrMatrix<T> sparse(2, 3, offsets, {}, {}, PinnedMemory{});
    const std::array<T, 3> values{1, 2, 3};
    Vector<T> x(values);
    expectValues<T>(sparse * x, {0, 0});
    expectSparse<T, int>(sparse + sparse, {0, 0, 0, 0, 0, 0});
    expectSparse<T, int>(la::transpose(sparse), {0, 0, 0, 0, 0, 0});
    CsrMatrix<T> right(3, 2, std::array<int, 4>{0, 0, 0, 0}, {}, {});
    expectSparse<T, int>(sparse * right, {0, 0, 0, 0});
    CsrMatrix<T> defaultEmpty;
    EXPECT_EQ((defaultEmpty + defaultEmpty).rowCount(), 0);
    EXPECT_EQ(la::transpose(defaultEmpty).rowCount(), 0);
    const std::array<int, 3> nonzeroOffsets{0, 1, 1};
    CsrMatrix<T> one(2, 3, nonzeroOffsets, std::array<int, 1>{1}, std::array<T, 1>{4});
    expectSparse<T, int>(sparse + one, {0, 4, 0, 0, 0, 0});
    expectSparse<T, int>(one + sparse, {0, 4, 0, 0, 0, 0});
    // 两个非零输入的乘积也可能没有任何结构非零项。
    CsrMatrix<T> disjoint(3, 2, std::array<int, 4>{0, 1, 1, 1}, std::array<int, 1>{0}, std::array<T, 1>{2});
    expectSparse<T, int>(one * disjoint, {0, 0, 0, 0});
}

/** @brief 验证形状、除零和尺寸范围错误在运算前被拒绝，无参数、无额外约束、无返回值。
 * @code ./build/test/thermofem_tests --gtest_filter=LinearAlgebraValidation.invalidShapes @endcode
 */
TEST(LinearAlgebraValidation, invalidShapes) {
    Vector<double> x, y;
    EXPECT_THROW(x / 0, std::domain_error);
    DenseMatrix<double> a(2, 0), b(3, 0);
    EXPECT_THROW(a + b, std::invalid_argument);
    EXPECT_THROW(a * b, std::invalid_argument);
    EXPECT_THROW(b * DenseMatrix<double>(1, 0), std::invalid_argument);
    EXPECT_THROW(la::detail::librarySize(static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1), std::length_error);
    CsrMatrix<double> sparse;
    EXPECT_THROW(sparse * b, std::invalid_argument);
}

/** @brief 验证运算设备切换及跨设备拒绝，无参数。@pre 两个设备可用。@return 无返回值。
 * @code ./build/test/thermofem_tests --gtest_filter=*deviceOwnership* @endcode
 */
TYPED_TEST(LinearAlgebraTest, deviceOwnership) {
    using T = TypeParam;
    int count{};
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    if (count < 2) { GTEST_SKIP() << "Two devices required"; }
    Vector<T> x(std::array<T, 2>{1, 2});
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    Vector<T> y(std::array<T, 2>{3, 4});
    EXPECT_THROW(x + y, std::invalid_argument);
    auto result = x * 2;
    EXPECT_EQ(result.deviceId(), 0);
    int current{};
    ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
    EXPECT_EQ(current, 1);
    expectValues<T>(result, {2, 4});
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
}
} // 匿名命名空间

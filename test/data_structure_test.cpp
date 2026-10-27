/** @file data_structure_test.cpp
 * @brief 验证存储策略传播、跨类型数据复制、矩阵不变量及非法输入处理。
 */
#include <DataStructure/CsrMatrix.cuh>
#include <DataStructure/DenseMatrix.cuh>
#include <DataStructure/Vector.cuh>

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

/** @brief 包含数据结构单元测试及测试辅助设施。 */
namespace {
    using namespace thermofem;

    static_assert(!std::is_base_of_v<CudaBuffer<double>, Vector<double>>);
    static_assert(!std::is_convertible_v<Vector<double>&, CudaBuffer<double>&>);
    static_assert(std::same_as<decltype(std::declval<Vector<double>&>().values()), const CudaBuffer<double>&>);
    static_assert(std::is_nothrow_move_constructible_v<Vector<double>>);
    static_assert(std::is_nothrow_move_assignable_v<Vector<double>>);
    static_assert(!BufferElement<std::string>);
    static_assert(!BufferElement<const double>);
    static_assert(!CsrIndex<bool>);
    static_assert(std::is_nothrow_move_constructible_v<CudaBuffer<double>>);
    static_assert(std::is_nothrow_move_assignable_v<DenseMatrix<double>>);
    static_assert(std::is_nothrow_move_assignable_v<CsrMatrix<double>>);

    /** @brief 构造测试策略。@param type 类型。@pre type 为已定义枚举值。@return 策略。
     * @code auto strategy = strategyFor(MemoryType::Global); @endcode
     */
    MemoryStrategy strategyFor(MemoryType type) {
        switch (type) {
        case MemoryType::Global:
            return GlobalMemory{};
        case MemoryType::Pinned:
            return PinnedMemory{};
        case MemoryType::Mapped:
            return MappedMemory{};
        case MemoryType::Managed:
            return ManagedMemory{};
        }
        throw std::invalid_argument("Unknown memory type");
    }

    /** @brief 检查测试设备是否可用。@param reason 不可用原因。@pre reason 有效。
     * @return 可用时为真。@code std::string reason; const bool ready = cudaAvailable(reason); @endcode
     */
    bool cudaAvailable(std::string& reason) {
        int count{};
        const auto result = cudaGetDeviceCount(&count);
        if (result != cudaSuccess || count == 0) {
            reason = result == cudaSuccess ? "No CUDA device" : cudaGetErrorString(result);
            return false;
        }
        const auto selected = cudaSetDevice(0);
        if (selected != cudaSuccess) {
            reason = cudaGetErrorString(selected);
            return false;
        }
        return true;
    }

    /** @brief 检查测试内存类型支持情况。@param type 类型。@pre 已选定设备 0。
     * @return 支持时为真。@code const bool supported = supports(MemoryType::Mapped); @endcode
     */
    bool supports(MemoryType type) {
        int result{};
        if (type == MemoryType::Mapped) {
            return cudaDeviceGetAttribute(&result, cudaDevAttrCanMapHostMemory, 0) == cudaSuccess && result;
        }
        if (type == MemoryType::Managed) {
            return cudaDeviceGetAttribute(&result, cudaDevAttrManagedMemory, 0) == cudaSuccess && result;
        }
        return true;
    }

    /** @brief 验证空缓冲区的类型传播和空指针行为，无参数、无额外约束、无返回值。
     * @code ./build/test/thermofem_tests --gtest_filter=BufferTest.emptyOwnership @endcode
     */
    TEST(BufferTest, emptyOwnership) {
        CudaBuffer<int> source(PinnedMemory{});
        CudaBuffer<int> target(MappedMemory{});
        target.copyDataFrom(source);
        EXPECT_EQ(target.memoryType(), MemoryType::Mapped);
        target = source;
        EXPECT_EQ(target.memoryType(), MemoryType::Pinned);
        target.changeMemoryType(ManagedMemory{});
        EXPECT_EQ(target.memoryType(), MemoryType::Managed);
        source = std::move(target);
        EXPECT_EQ(source.memoryType(), MemoryType::Managed);
        EXPECT_TRUE(target.empty());
        EXPECT_EQ(source.hostData(), nullptr);
        EXPECT_EQ(source.deviceData(), nullptr);
        source.copyFromHost({});
        source.copyToHost({});
        source = source;
        auto& sourceAlias = source;
        source = std::move(sourceAlias);
        EXPECT_TRUE(source.empty());
    }

    /** @brief 验证分配前的字节数溢出检查，无参数、无额外约束、无返回值。
     * @code ./build/test/thermofem_tests --gtest_filter=BufferTest.rejectsSizeOverflow @endcode
     */
    TEST(BufferTest, rejectsSizeOverflow) {
        const auto count = std::numeric_limits<std::size_t>::max() / sizeof(double) + 1;
        EXPECT_THROW((CudaBuffer<double>(count)), std::length_error);
        EXPECT_THROW((DenseMatrix<double>(std::numeric_limits<std::size_t>::max(), 2)), std::length_error);
    }

    /** @brief 验证空缓冲区拒绝错误的主机数组长度，无参数、无额外约束、无返回值。
     * @code ./build/test/thermofem_tests --gtest_filter=BufferTest.rejectsHostSizeMismatch @endcode
     */
    TEST(BufferTest, rejectsHostSizeMismatch) {
        CudaBuffer<int> buffer;
        std::array<int, 1> values{1};
        EXPECT_THROW(buffer.copyFromHost(values), std::invalid_argument);
        EXPECT_THROW(buffer.copyToHost(values), std::invalid_argument);
    }

    /** @brief 验证空矩阵移动后的维度以及形状校验，无参数、无额外约束、无返回值。
     * @code ./build/test/thermofem_tests --gtest_filter=DenseMatrixTest.emptyShapeAndMove @endcode
     */
    TEST(DenseMatrixTest, emptyShapeAndMove) {
        DenseMatrix<double> source(3, 0, PinnedMemory{});
        DenseMatrix<double> target(std::move(source));
        EXPECT_EQ(source.rowCount(), 0);
        EXPECT_EQ(source.columnCount(), 0);
        EXPECT_EQ(target.rowCount(), 3);
        EXPECT_EQ(target.memoryType(), MemoryType::Pinned);
        DenseMatrix<double> different(4, 0);
        EXPECT_THROW(target.copyDataFrom(different), std::invalid_argument);
        EXPECT_THROW(target.at(0, 0), std::out_of_range);
        target = target;
        auto& targetAlias = target;
        target = std::move(targetAlias);
        EXPECT_EQ(target.rowCount(), 3);
    }

    /** @brief 验证 CSR 数组长度与端点错误在分配前被拒绝，无参数、无额外约束、无返回值。
     * @code ./build/test/thermofem_tests --gtest_filter=CsrMatrixTest.rejectsInvalidSizesAndEndpoints @endcode
     */
    TEST(CsrMatrixTest, rejectsInvalidSizesAndEndpoints) {
        const std::array<int, 3> offsets{0, 1, 2};
        const std::array<int, 2> columns{0, 1};
        const std::array<double, 2> values{2, 3};
        EXPECT_THROW((CsrMatrix<double>(2, 2, {}, columns, values)), std::invalid_argument);
        EXPECT_THROW((CsrMatrix<double>(3, 2, offsets, columns, values)), std::invalid_argument);
        EXPECT_THROW((CsrMatrix<double>(2, 2, offsets, columns, {})), std::invalid_argument);
        const std::array<int, 3> badStart{1, 1, 2};
        EXPECT_THROW((CsrMatrix<double>(2, 2, badStart, columns, values)), std::invalid_argument);
        const std::array<int, 3> badEnd{0, 0, 1};
        EXPECT_THROW((CsrMatrix<double>(2, 2, badEnd, columns, values)), std::invalid_argument);
    }

    /** @brief 验证 CSR 偏移和列索引的合法性检查，无参数、无额外约束、无返回值。
     * @code ./build/test/thermofem_tests --gtest_filter=CsrMatrixTest.rejectsInvalidIndices @endcode
     */
    TEST(CsrMatrixTest, rejectsInvalidIndices) {
        const std::array<int, 4> decreasing{0, 2, 1, 2};
        const std::array<int, 4> negative{0, -1, 1, 2};
        const std::array<int, 4> outOfBounds{0, 3, 1, 2};
        const std::array<int, 2> columns{0, 1};
        const std::array<double, 2> values{2, 3};
        EXPECT_THROW((CsrMatrix<double>(3, 2, decreasing, columns, values)), std::invalid_argument);
        EXPECT_THROW((CsrMatrix<double>(3, 2, negative, columns, values)), std::invalid_argument);
        EXPECT_THROW((CsrMatrix<double>(3, 2, outOfBounds, columns, values)), std::invalid_argument);
        const std::array<int, 3> offsets{0, 1, 2};
        const std::array<int, 2> badColumns{-1, 1};
        const std::array<int, 2> oversizedColumns{0, 2};
        EXPECT_THROW((CsrMatrix<double>(2, 2, offsets, badColumns, values)), std::invalid_argument);
        EXPECT_THROW((CsrMatrix<double>(2, 2, offsets, oversizedColumns, values)), std::invalid_argument);
    }

    /** @brief 验证 CSR 索引类型范围检查和默认空对象，无参数、无额外约束、无返回值。
     * @code ./build/test/thermofem_tests --gtest_filter=CsrMatrixTest.indexOverflowAndEmptyMove @endcode
     */
    TEST(CsrMatrixTest, indexOverflowAndEmptyMove) {
        EXPECT_THROW((CsrMatrix<double>(std::numeric_limits<std::size_t>::max(), 2, {}, {}, {})), std::length_error);
        CsrMatrix<double> source;
        source.changeMemoryType(PinnedMemory{});
        CsrMatrix<double> target(std::move(source));
        EXPECT_EQ(target.rowCount(), 0);
        EXPECT_EQ(target.nonZeroCount(), 0);
        EXPECT_EQ(target.rowOffsets().memoryType(), MemoryType::Pinned);
        EXPECT_EQ(target.columnIndices().memoryType(), MemoryType::Pinned);
        EXPECT_EQ(source.rowCount(), 0);
        target = target;
        auto& targetAlias = target;
        target = std::move(targetAlias);
    }

    /** @brief 为全部源、目标内存类型组合提供真实 CUDA 测试环境。
     * @code ./build/test/thermofem_tests --gtest_filter=allMemoryPairs* @endcode
     */
    class CudaTransferTest : public ::testing::TestWithParam<std::tuple<MemoryType, MemoryType>> {
    protected:
        /** @brief 初始化设备并跳过不支持的类型，无参数、无额外约束、无返回值。
         * @code SetUp(); @endcode
         */
        void SetUp() override {
            std::string reason;
            if (!cudaAvailable(reason)) {
                GTEST_SKIP() << reason;
            }
            if (!supports(std::get<0>(GetParam())) || !supports(std::get<1>(GetParam()))) {
                GTEST_SKIP() << "Requested memory policy is not supported by this device";
            }
        }
    };

    /** @brief 验证跨类型只复制数据与深拷贝赋值，无参数。@pre CUDA 可用。@return 无返回值。
     * @code ./build/test/thermofem_tests --gtest_filter=*copyDataAndAssignment* @endcode
     */
    TEST_P(CudaTransferTest, copyDataAndAssignment) {
        const auto [sourceType, targetType] = GetParam();
        const std::array<int, 4> expected{1, -2, 3, 7};
        const std::array<int, 4> changed{9, 8, 7, 6};
        CudaBuffer<int> source(expected, strategyFor(sourceType));
        CudaBuffer<int> target(4, strategyFor(targetType));
        auto* previousPointer = target.data();
        target.copyDataFrom(source);
        EXPECT_EQ(target.data(), previousPointer);
        EXPECT_EQ(target.memoryType(), targetType);
        std::array<int, 4> actual{};
        target.copyToHost(actual);
        EXPECT_EQ(actual, expected);
        target = source;
        EXPECT_EQ(target.memoryType(), sourceType);
        EXPECT_NE(target.data(), source.data());
        source.copyFromHost(changed);
        target.copyToHost(actual);
        EXPECT_EQ(actual, expected);
        CudaBuffer<int> wrongSize(1, strategyFor(targetType));
        EXPECT_THROW(wrongSize.copyDataFrom(source), std::invalid_argument);
        target = target;
        target.copyDataFrom(target);
        target.copyToHost(actual);
        EXPECT_EQ(actual, expected);
    }

    /** @brief 验证转换保留数据、移动接管所有权及主机设备视图，无参数。
     * @pre CUDA 可用。@return 无返回值。
     * @code ./build/test/thermofem_tests --gtest_filter=*conversionAndMove* @endcode
     */
    TEST_P(CudaTransferTest, conversionAndMove) {
        const auto [sourceType, targetType] = GetParam();
        const std::array<int, 4> expected{3, 1, 4, 1};
        CudaBuffer<int> source(expected, strategyFor(sourceType));
        source.changeMemoryType(strategyFor(targetType));
        EXPECT_EQ(source.memoryType(), targetType);
        std::array<int, 4> actual{};
        source.copyToHost(actual);
        EXPECT_EQ(actual, expected);
        if (targetType == MemoryType::Global) {
            EXPECT_THROW((void)source.hostData(), std::logic_error);
        }
        else {
            EXPECT_EQ(source.hostData()[0], expected[0]);
        }
        if (targetType == MemoryType::Pinned) {
            EXPECT_THROW((void)source.deviceData(), std::logic_error);
        }
        else {
            EXPECT_NE(source.deviceData(), nullptr);
        }
        auto* previousPointer = source.data();
        CudaBuffer<int> moved(std::move(source));
        EXPECT_EQ(moved.data(), previousPointer);
        EXPECT_TRUE(source.empty());
        CudaBuffer<int> target(1, strategyFor(sourceType));
        target = std::move(moved);
        EXPECT_EQ(target.data(), previousPointer);
        EXPECT_EQ(target.memoryType(), targetType);
        EXPECT_TRUE(moved.empty());
        auto& targetAlias = target;
        target = std::move(targetAlias);
        target.copyToHost(actual);
        EXPECT_EQ(actual, expected);
    }

    INSTANTIATE_TEST_SUITE_P(allMemoryPairs, CudaTransferTest,
                             ::testing::Combine(::testing::Values(MemoryType::Global, MemoryType::Pinned,
                                                                  MemoryType::Mapped, MemoryType::Managed),
                                                ::testing::Values(MemoryType::Global, MemoryType::Pinned,
                                                                  MemoryType::Mapped, MemoryType::Managed)));

    /** @brief 验证组合向量在全部内存策略配对下的所有权与复制语义，无参数。
     * @pre CUDA 可用。@return 无返回值。
     * @code ./build/test/thermofem_tests --gtest_filter=VectorTest.* @endcode
     */
    TEST(VectorTest, composedStoragePreservesOwnershipAndPolicy) {
        std::string reason;
        if (!cudaAvailable(reason)) {
            GTEST_SKIP() << reason;
        }
        const std::array<double, 3> expected{1, 2, 3}, changed{4, 5, 6};
        for (auto sourceType : {MemoryType::Global, MemoryType::Pinned, MemoryType::Mapped, MemoryType::Managed}) {
            if (!supports(sourceType)) {
                continue;
            }
            for (auto targetType : {MemoryType::Global, MemoryType::Pinned, MemoryType::Mapped, MemoryType::Managed}) {
                if (!supports(targetType)) {
                    continue;
                }
                Vector<double> source(expected, strategyFor(sourceType));
                Vector<double> target(3, strategyFor(targetType));
                auto* originalAddress = target.data();
                target.copyDataFrom(source);
                EXPECT_EQ(target.data(), originalAddress);
                EXPECT_EQ(target.memoryType(), targetType);
                std::array<double, 3> actual{};
                target.copyToHost(actual);
                EXPECT_EQ(actual, expected);
                target = source;
                EXPECT_EQ(target.memoryType(), sourceType);
                EXPECT_NE(target.data(), source.data());
                source.copyFromHost(changed);
                target.copyToHost(actual);
                EXPECT_EQ(actual, expected);
                auto* transferredAddress = target.data();
                Vector<double> moved(std::move(target));
                EXPECT_TRUE(target.empty());
                EXPECT_EQ(moved.data(), transferredAddress);
                target = std::move(moved);
                EXPECT_TRUE(moved.empty());
                EXPECT_EQ(target.data(), transferredAddress);
                target.changeMemoryType(strategyFor(targetType));
                EXPECT_EQ(target.memoryType(), targetType);
                target.copyToHost(actual);
                EXPECT_EQ(actual, expected);
                target.copyDataFrom(source.values());
                target.copyToHost(actual);
                EXPECT_EQ(actual, changed);
                target.swap(source);
                EXPECT_EQ(source.memoryType(), targetType);
                EXPECT_EQ(target.memoryType(), sourceType);
                Vector<double> wrong(2);
                EXPECT_THROW(target.copyDataFrom(wrong), std::invalid_argument);
            }
        }
    }

    /** @brief 验证稠密矩阵布局与所有权，无参数。@pre CUDA 可用。@return 无返回值。
     * @code ./build/test/thermofem_tests --gtest_filter=DenseMatrixTest.storageAndValueSemantics @endcode */
    TEST(DenseMatrixTest, storageAndValueSemantics) {
        std::string reason;
        if (!cudaAvailable(reason)) {
            GTEST_SKIP() << reason;
        }
        const std::array<double, 6> expected{1, 2, 3, 4, 5, 6};
        Vector<double> vector(expected, PinnedMemory{});
        Vector<double> vectorCopy = vector;
        vector.hostData()[0] = -1;
        EXPECT_EQ(vectorCopy.hostData()[0], 1);
        DenseMatrix<double> source(2, 3, PinnedMemory{});
        source.copyFromHost(expected);
        EXPECT_EQ(source.at(1, 2), 6);
        EXPECT_THROW(source.at(2, 0), std::out_of_range);
        DenseMatrix<double> target(2, 3);
        target.copyDataFrom(source);
        EXPECT_EQ(target.memoryType(), MemoryType::Global);
        target.changeMemoryType(PinnedMemory{});
        EXPECT_EQ(target.at(1, 2), 6);
        target = source;
        source.at(0, 0) = -1;
        EXPECT_EQ(target.at(0, 0), 1);
        DenseMatrix<double> moved(std::move(target));
        EXPECT_EQ(target.rowCount(), 0);
        EXPECT_EQ(target.values().size(), 0);
        EXPECT_EQ(moved.at(1, 2), 6);
    }

    /** @brief 验证 CSR 三缓冲区的一致性及深拷贝，无参数。@pre CUDA 可用。@return 无返回值。
     * @code ./build/test/thermofem_tests --gtest_filter=CsrMatrixTest.storageAndValueSemantics @endcode
     */
    TEST(CsrMatrixTest, storageAndValueSemantics) {
        std::string reason;
        if (!cudaAvailable(reason)) {
            GTEST_SKIP() << reason;
        }
        const std::array<std::int64_t, 4> offsets{0, 2, 2, 3};
        const std::array<std::int64_t, 3> columns{2, 0, 1};
        const std::array<double, 3> values{2, 3, 4};
        CsrMatrix<double, std::int64_t> source(3, 3, offsets, columns, values, PinnedMemory{});
        auto target = source;
        target.changeMemoryType(GlobalMemory{});
        EXPECT_EQ(target.rowOffsets().memoryType(), MemoryType::Global);
        EXPECT_EQ(target.columnIndices().memoryType(), MemoryType::Global);
        std::array<std::int64_t, 4> actualOffsets{};
        std::array<std::int64_t, 3> actualColumns{};
        std::array<double, 3> actualValues{};
        target.rowOffsets().copyToHost(actualOffsets);
        target.columnIndices().copyToHost(actualColumns);
        target.values().copyToHost(actualValues);
        EXPECT_EQ(actualOffsets, offsets);
        EXPECT_EQ(actualColumns, columns);
        EXPECT_EQ(actualValues, values);
        source.copyValuesFromHost(std::array<double, 3>{9, 8, 7});
        target.values().copyToHost(actualValues);
        EXPECT_EQ(actualValues, values);
        target = source;
        EXPECT_EQ(target.memoryType(), MemoryType::Pinned);
        CsrMatrix<double, std::int64_t> moved(std::move(target));
        EXPECT_EQ(moved.nonZeroCount(), 3);
        EXPECT_EQ(target.rowCount(), 0);
        EXPECT_TRUE(target.rowOffsets().empty());
        const std::array<int, 3> emptyOffsets{0, 0, 0};
        CsrMatrix<double> emptyRows(2, 4, emptyOffsets, {}, {}, PinnedMemory{});
        EXPECT_EQ(emptyRows.rowCount(), 2);
        EXPECT_EQ(emptyRows.nonZeroCount(), 0);
    }

    /** @brief 验证设备归属、上下文恢复以及跨设备复制的拒绝行为，无参数。
     * @pre 至少两个 CUDA 设备可用。@return 无返回值。
     * @code ./build/test/thermofem_tests --gtest_filter=BufferTest.deviceOwnership @endcode
     */
    TEST(BufferTest, deviceOwnership) {
        std::string reason;
        if (!cudaAvailable(reason)) {
            GTEST_SKIP() << reason;
        }
        int count{};
        if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
            GTEST_SKIP() << "At least two CUDA devices are required";
        }
        const std::array<int, 2> expected{3, 7};
        CudaBuffer<int> source(expected);
        ASSERT_EQ(source.deviceId(), 0);
        ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
        {
            CudaBuffer<int> target(2);
            EXPECT_EQ(target.deviceId(), 1);
            EXPECT_THROW(target.copyDataFrom(source), std::invalid_argument);
            auto copy = source;
            EXPECT_EQ(copy.deviceId(), 0);
            int currentDevice{};
            ASSERT_EQ(cudaGetDevice(&currentDevice), cudaSuccess);
            EXPECT_EQ(currentDevice, 1);
            source.changeMemoryType(PinnedMemory{});
            EXPECT_EQ(source.deviceId(), 0);
            std::array<int, 2> actual{};
            source.copyToHost(actual);
            EXPECT_EQ(actual, expected);
            ASSERT_EQ(cudaGetDevice(&currentDevice), cudaSuccess);
            EXPECT_EQ(currentDevice, 1);
        }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }
} // namespace

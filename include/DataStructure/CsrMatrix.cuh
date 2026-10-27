/** @file CsrMatrix.cuh
 * @brief 定义按行压缩的 CUDA 稀疏矩阵及其结构校验。
 */
#pragma once

#include <DataStructure/CudaBuffer.cuh>

#include <cstdint>
#include <limits>

namespace thermofem {
    namespace detail {
        /** @brief 在设备端验证 CSR 结构，仅回传错误标志。
         * @param rows 行数。@param columns 列数。@param count 非零数。@param offsets 行偏移。
         * @param indices 列索引。@param wide 是否使用 64 位索引。@param device 所属设备。@pre 地址有效。
         * @note 无返回值，非法结构抛异常。@code validateDeviceCsr(n,n,nz,rp,ci,false,device); @endcode */
        void validateDeviceCsr(std::size_t rows, std::size_t columns, std::size_t count, const void* offsets,
                               const void* indices, bool wide, int device);
    } // namespace detail
    /// 限定 CSR 索引为 32 位或 64 位有符号整数。
    template <class T>
    concept CsrIndex = std::signed_integral<T> && (sizeof(T) == 4 || sizeof(T) == 8);

    /** @brief 使用独立缓冲区保存行偏移、列索引与非零元素的 CSR 矩阵。
     * @tparam T 满足 BufferElement 的元素类型。@tparam Index 满足 CsrIndex 的索引类型。
     * @details 构造时检查 CSR 结构；允许未排序的列索引、重复列索引及显式零元素。
     * 索引接口只读以保护结构，数值可单独更新。三个缓冲区始终具有相同内存策略。
     * 默认及移动后对象为无存储的 0×0 空矩阵；正常构造的行偏移长度为行数加一。
     * @code
     * const int offsets[]{0, 1, 2};
     * const int columns[]{0, 1};
     * const double entries[]{2.0, 3.0};
     * thermofem::CsrMatrix<double> matrix(2, 2, offsets, columns, entries);
     * @endcode
     */
    template <BufferElement T, CsrIndex Index = std::int32_t>
    class CsrMatrix {
    public:
        using ValueType = T; ///< 非零元素类型。
        /** @brief 创建无存储的空矩阵，无参数、无额外约束。@note 无返回值。
         * @code thermofem::CsrMatrix<double> matrix; @endcode
         */
        CsrMatrix() = default;
        /** @brief 验证并接管 GPU 算法生成的三个 global 缓冲区，避免下载整个结构。
         * @param rows 行数。@param columns 列数。@param offsets 行偏移所有权。
         * @param indices 列索引所有权。@param values 数值所有权。
         * @pre 缓冲区属于同一设备且为 global；结构合法，尺寸不超出 Index。
         * @return 拥有这些缓冲区的矩阵；校验失败不移动输入。
         * @details 在所属 GPU 校验偏移和列索引，允许重复、无序与显式零。
         * 不检查数值有限性，这与主机构造接口一致。调用完成后恢复调用方设备。
         * @code auto a = CsrMatrix<double>::fromBuffers(n,n,std::move(rp),std::move(ci),std::move(v)); @endcode */
        static CsrMatrix fromBuffers(std::size_t rows, std::size_t columns, CudaBuffer<Index>&& offsets,
                                     CudaBuffer<Index>&& indices, CudaBuffer<T>&& values) {
            if (!std::in_range<Index>(rows) || !std::in_range<Index>(columns) || !std::in_range<Index>(values.size()) ||
                rows == std::numeric_limits<std::size_t>::max()) {
                throw std::length_error("CSR dimensions exceed index range");
            }
            if (offsets.size() != rows + 1 || indices.size() != values.size() ||
                offsets.memoryType() != MemoryType::Global || indices.memoryType() != MemoryType::Global ||
                values.memoryType() != MemoryType::Global ||
                (!indices.empty() && offsets.deviceId() != indices.deviceId()) ||
                (!values.empty() && offsets.deviceId() != values.deviceId())) {
                throw std::invalid_argument("Invalid CSR device buffers");
            }
            detail::validateDeviceCsr(rows, columns, values.size(), offsets.deviceData(), indices.deviceData(),
                                      sizeof(Index) == 8, offsets.deviceId());
            CsrMatrix result;
            result.rowCount_ = rows;
            result.columnCount_ = columns;
            result.rowOffsets_ = std::move(offsets);
            result.columnIndices_ = std::move(indices);
            result.values_ = std::move(values);
            return result;
        }
        /** @brief 校验主机 CSR 数据并按策略复制到内部存储。
         * @param rows 行数。@param columns 列数。@param rowOffsets 行偏移数组。
         * @param columnIndices 列索引数组。@param values 非零元素数组。@param strategy 内存策略。
         * @pre 偏移从 0 开始单调不减且末项等于非零数，列索引合法，尺寸可由 Index 表示。
         * @note 无返回值。@code thermofem::CsrMatrix<double> matrix(2, 2, offsets, columns, entries); @endcode
         */
        CsrMatrix(std::size_t rows, std::size_t columns, std::span<const Index> rowOffsets,
                  std::span<const Index> columnIndices, std::span<const T> values,
                  MemoryStrategy strategy = GlobalMemory{}) : rowCount_(rows), columnCount_(columns) {
            validate(rows, columns, rowOffsets, columnIndices, values.size());
            rowOffsets_ = CudaBuffer<Index>(rowOffsets, strategy);
            columnIndices_ = CudaBuffer<Index>(columnIndices, strategy);
            values_ = CudaBuffer<T>(values, strategy);
        }
        /** @brief 深拷贝全部数据及策略。@param other 来源。@pre 异步访问已结束。@note 无返回值。
         * @code auto copy = matrix; @endcode
         */
        CsrMatrix(const CsrMatrix& other) = default;
        /** @brief 接管来源并清空其维度。@param other 来源。@pre 无额外约束。@note 无返回值。
         * @code auto moved = std::move(matrix); @endcode
         */
        CsrMatrix(CsrMatrix&& other) noexcept { swap(other); }
        /** @brief 以强异常保证复制结构、数值及策略。@param other 来源。@pre 异步访问已结束。
         * @return 当前对象。@code target = source; @endcode
         */
        CsrMatrix& operator=(const CsrMatrix& other) {
            if (this != &other) {
                CsrMatrix temporary(other);
                swap(temporary);
            }
            return *this;
        }
        /** @brief 接管来源并释放旧存储。@param other 来源。@pre 异步访问已结束。
         * @return 当前对象。@code target = std::move(source); @endcode
         */
        CsrMatrix& operator=(CsrMatrix&& other) noexcept {
            if (this != &other) {
                CsrMatrix temporary(std::move(other));
                swap(temporary);
            }
            return *this;
        }
        /** @brief 交换维度、策略及三个缓冲区。@param other 交换对象。@pre 无额外约束。
         * @note 无返回值。@code target.swap(source); @endcode
         */
        void swap(CsrMatrix& other) noexcept {
            std::swap(rowCount_, other.rowCount_);
            std::swap(columnCount_, other.columnCount_);
            rowOffsets_.swap(other.rowOffsets_);
            columnIndices_.swap(other.columnIndices_);
            values_.swap(other.values_);
        }
        /** @brief 保留矩阵内容并原地转换全部缓冲区，失败不改变原对象。
         * @param strategy 新策略。@pre 异步访问已结束。@note 无返回值。
         * @code matrix.changeMemoryType(thermofem::PinnedMemory{}); @endcode
         */
        void changeMemoryType(MemoryStrategy strategy) {
            if (values_.memoryStrategy().index() == strategy.index()) {
                return;
            }
            CsrMatrix temporary(*this);
            temporary.rowOffsets_.changeMemoryType(strategy);
            temporary.columnIndices_.changeMemoryType(strategy);
            temporary.values_.changeMemoryType(strategy);
            swap(temporary);
        }
        /** @brief 从主机更新非零元素，保持稀疏结构不变。@param values 新数值。
         * @pre 数量等于非零数且异步访问已结束。@note 无返回值。
         * @code matrix.copyValuesFromHost(entries); @endcode
         */
        void copyValuesFromHost(std::span<const T> values) { values_.copyFromHost(values); }
        /** @brief 查询行数，无参数、无额外约束。@return 行数。
         * @code const auto rows = matrix.rowCount(); @endcode
         */
        [[nodiscard]] std::size_t rowCount() const noexcept { return rowCount_; }
        /** @brief 查询列数，无参数、无额外约束。@return 列数。
         * @code const auto columns = matrix.columnCount(); @endcode
         */
        [[nodiscard]] std::size_t columnCount() const noexcept { return columnCount_; }
        /** @brief 查询存储的非零数，无参数、无额外约束。@return 非零数，含显式零。
         * @code const auto count = matrix.nonZeroCount(); @endcode
         */
        [[nodiscard]] std::size_t nonZeroCount() const noexcept { return values_.size(); }
        /** @brief 查询类型，无参数、无额外约束。@return 三个缓冲区共同的内存类别。
         * @code const auto type = matrix.memoryType(); @endcode
         */
        [[nodiscard]] MemoryType memoryType() const noexcept { return values_.memoryType(); }
        /** @brief 获取只读行偏移接口，无参数、无额外约束。@return 行偏移缓冲区常量引用。
         * @code const auto* offsets = matrix.rowOffsets().deviceData(); @endcode
         */
        [[nodiscard]] const CudaBuffer<Index>& rowOffsets() const noexcept { return rowOffsets_; }
        /** @brief 获取只读列索引接口，无参数、无额外约束。@return 列索引缓冲区常量引用。
         * @code const auto* columns = matrix.columnIndices().deviceData(); @endcode
         */
        [[nodiscard]] const CudaBuffer<Index>& columnIndices() const noexcept { return columnIndices_; }
        /** @brief 获取只读数值接口，无参数、无额外约束。@return 数值缓冲区常量引用。
         * @code matrix.values().copyToHost(entries); @endcode
         */
        [[nodiscard]] const CudaBuffer<T>& values() const noexcept { return values_; }
        /** @brief 获取可写设备数值地址，无参数。@pre 非空 pinned 不可调用，使用所属设备。
         * @return 设备数值地址。@code auto* entries = matrix.deviceValues(); @endcode
         */
        [[nodiscard]] T* deviceValues() { return values_.deviceData(); }
        /** @brief 获取只读设备数值地址，无参数。@pre 同可写版本。@return 只读设备地址。
         * @code const auto* entries = std::as_const(matrix).deviceValues(); @endcode
         */
        [[nodiscard]] const T* deviceValues() const { return values_.deviceData(); }

    private:
        /** @brief 在分配前检查主机 CSR 输入。
         * @param rows 行数。@param columns 列数。@param offsets 行偏移。@param indices 列索引。
         * @param count 非零数。@pre 无额外约束，非法输入抛异常。@note 无返回值。
         * @code validate(2, 2, offsets, columns, 2); @endcode
         */
        static void validate(std::size_t rows, std::size_t columns, std::span<const Index> offsets,
                             std::span<const Index> indices, std::size_t count) {
            if (!std::in_range<Index>(rows) || !std::in_range<Index>(columns) || !std::in_range<Index>(count) ||
                rows == std::numeric_limits<std::size_t>::max()) {
                throw std::length_error("CSR dimensions exceed the index range");
            }
            if (offsets.size() != rows + 1 || indices.size() != count || offsets.front() != 0 ||
                offsets.back() != static_cast<Index>(count)) {
                throw std::invalid_argument("Invalid CSR array sizes or endpoints");
            }
            Index previous{};
            for (const auto offset : offsets) {
                if (offset < previous || offset > static_cast<Index>(count)) {
                    throw std::invalid_argument("CSR row offsets must be nondecreasing and bounded");
                }
                previous = offset;
            }
            for (const auto column : indices) {
                if (column < 0 || static_cast<std::size_t>(column) >= columns) {
                    throw std::invalid_argument("CSR column index is out of range");
                }
            }
        }
        std::size_t rowCount_{}; ///< 矩阵行数。
        std::size_t columnCount_{}; ///< 矩阵列数。
        CudaBuffer<Index> rowOffsets_; ///< 每行起点及最后一行终点。
        CudaBuffer<Index> columnIndices_; ///< 每个非零元素的列索引。
        CudaBuffer<T> values_; ///< 与列索引逐项对应的非零元素值。
    };
} // namespace thermofem

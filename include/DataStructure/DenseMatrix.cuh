/** @file DenseMatrix.cuh
 * @brief 定义行主序、连续存储的 CUDA 稠密矩阵。
 */
#pragma once

#include <DataStructure/CudaBuffer.cuh>

namespace thermofem {
    /** @brief 按行主序存储稠密矩阵，并保持维度与缓冲区大小一致。
     * @tparam T 满足 BufferElement 的元素类型。
     * @details 索引 (row, column) 对应 row * columnCount() + column。
     * 复制及移动采用来源策略；移动后的来源为 0 行 0 列。
     * @code
     * thermofem::DenseMatrix<double> matrix(2, 3, thermofem::PinnedMemory{});
     * matrix.at(0, 1) = 2.0;
     * matrix.changeMemoryType(thermofem::GlobalMemory{});
     * @endcode
     */
    template <BufferElement T>
    class DenseMatrix {
    public:
        using ValueType = T; ///< 矩阵元素类型。
        /** @brief 创建空矩阵，无参数、无额外约束。@note 无返回值。
         * @code thermofem::DenseMatrix<double> matrix; @endcode
         */
        DenseMatrix() = default;
        /** @brief 分配未初始化矩阵。@param rows 行数。@param columns 列数。@param strategy 内存策略。
         * @pre 元素数和字节数不溢出，非空分配需要 CUDA。@note 无返回值。
         * @code thermofem::DenseMatrix<double> matrix(2, 3); @endcode
         */
        DenseMatrix(std::size_t rows, std::size_t columns, MemoryStrategy strategy = GlobalMemory{}) :
            rowCount_(rows), columnCount_(columns), values_(detail::checkedProduct(rows, columns), strategy) {}
        /** @brief 深拷贝矩阵及策略。@param other 来源。@pre 异步访问已结束。@note 无返回值。
         * @code auto copy = matrix; @endcode
         */
        DenseMatrix(const DenseMatrix& other) = default;
        /** @brief 接管矩阵并清空来源。@param other 来源。@pre 无额外约束。@note 无返回值。
         * @code auto moved = std::move(matrix); @endcode
         */
        DenseMatrix(DenseMatrix&& other) noexcept { swap(other); }
        /** @brief 以强异常保证复制形状、数据与策略。@param other 来源。@pre 异步访问已结束。
         * @return 当前对象。@code target = source; @endcode
         */
        DenseMatrix& operator=(const DenseMatrix& other) {
            if (this != &other) {
                DenseMatrix temporary(other);
                swap(temporary);
            }
            return *this;
        }
        /** @brief 接管来源并释放旧矩阵。@param other 来源。@pre 异步访问已结束。
         * @return 当前对象。@code target = std::move(source); @endcode
         */
        DenseMatrix& operator=(DenseMatrix&& other) noexcept {
            if (this != &other) {
                DenseMatrix temporary(std::move(other));
                swap(temporary);
            }
            return *this;
        }
        /** @brief 交换矩阵形状、策略与所有权。@param other 交换对象。@pre 无额外约束。
         * @note 无返回值。@code target.swap(source); @endcode
         */
        void swap(DenseMatrix& other) noexcept {
            std::swap(rowCount_, other.rowCount_);
            std::swap(columnCount_, other.columnCount_);
            values_.swap(other.values_);
        }
        /** @brief 保留形状、类型和地址，复制全部元素。@param other 来源。
         * @pre 行列数一致、设备一致且异步访问已结束。@note 无返回值。
         * @code target.copyDataFrom(source); @endcode
         */
        void copyDataFrom(const DenseMatrix& other) {
            if (rowCount_ != other.rowCount_ || columnCount_ != other.columnCount_) {
                throw std::invalid_argument("Dense matrix shapes must match");
            }
            values_.copyDataFrom(other.values_);
        }
        /** @brief 保留形状与数据，转换内存类型。@param strategy 新策略。@pre 异步访问已结束。
         * @note 无返回值。@code matrix.changeMemoryType(thermofem::MappedMemory{}); @endcode
         */
        void changeMemoryType(MemoryStrategy strategy) { values_.changeMemoryType(strategy); }
        /** @brief 从主机按行主序导入全部元素。@param values 主机数组。@pre 元素数匹配且异步访问已结束。
         * @note 无返回值。@code matrix.copyFromHost(values); @endcode
         */
        void copyFromHost(std::span<const T> values) { values_.copyFromHost(values); }
        /** @brief 从缓冲区导入全部元素，保留矩阵形状、策略与地址。@param values 来源。
         * @pre 元素数相同且设备一致，异步访问已结束。@note 无返回值。
         * @code matrix.copyValuesFrom(buffer); @endcode
         */
        void copyValuesFrom(const CudaBuffer<T>& values) { values_.copyDataFrom(values); }
        /** @brief 按行主序导出至主机。@param values 主机数组。@pre 元素数匹配且异步访问已结束。
         * @note 无返回值。@code matrix.copyToHost(values); @endcode
         */
        void copyToHost(std::span<T> values) const { values_.copyToHost(values); }
        /** @brief 查询行数，无参数、无额外约束。@return 行数。
         * @code const auto rows = matrix.rowCount(); @endcode
         */
        [[nodiscard]] std::size_t rowCount() const noexcept { return rowCount_; }
        /** @brief 查询列数，无参数、无额外约束。@return 列数。
         * @code const auto columns = matrix.columnCount(); @endcode
         */
        [[nodiscard]] std::size_t columnCount() const noexcept { return columnCount_; }
        /** @brief 获取只读存储接口，无参数、无额外约束。@return 元素缓冲区常量引用。
         * @code const auto count = matrix.values().size(); @endcode
         */
        [[nodiscard]] const CudaBuffer<T>& values() const noexcept { return values_; }
        /** @brief 查询内存类型，无参数、无额外约束。@return 内存类别。
         * @code const auto type = matrix.memoryType(); @endcode
         */
        [[nodiscard]] MemoryType memoryType() const noexcept { return values_.memoryType(); }
        /** @brief 获取设备元素地址，无参数。@pre 遵循 CudaBuffer::deviceData() 的要求。
         * @return 设备地址。@code auto* pointer = matrix.deviceData(); @endcode
         */
        [[nodiscard]] T* deviceData() { return values_.deviceData(); }
        /** @brief 获取只读设备地址，无参数。@pre 同可写版本。@return 只读设备地址。
         * @code const auto* pointer = std::as_const(matrix).deviceData(); @endcode
         */
        [[nodiscard]] const T* deviceData() const { return values_.deviceData(); }
        /** @brief 获取主机上的指定元素。@param row 行索引。@param column 列索引。
         * @pre 索引有效且内存可由主机访问，异步访问已结束；否则抛异常。@return 元素引用。
         * @code matrix.at(0, 1) = 2.0; @endcode
         */
        T& at(std::size_t row, std::size_t column) { return const_cast<T&>(std::as_const(*this).at(row, column)); }
        /** @brief 获取主机上的只读元素。@param row 行索引。@param column 列索引。
         * @pre 同可写版本。@return 元素常量引用。@code const auto value = std::as_const(matrix).at(0, 1); @endcode
         */
        const T& at(std::size_t row, std::size_t column) const {
            if (row >= rowCount_ || column >= columnCount_) {
                throw std::out_of_range("Dense matrix index is out of range");
            }
            return values_.hostData()[row * columnCount_ + column];
        }

    private:
        std::size_t rowCount_{}; ///< 矩阵行数。
        std::size_t columnCount_{}; ///< 矩阵列数。
        CudaBuffer<T> values_; ///< 行主序元素存储。
    };
} // namespace thermofem

/** @file Vector.cuh
 * @brief 定义组合持有 CUDA 缓冲区的一维向量。
 */
#pragma once

#include <DataStructure/CudaBuffer.cuh>

namespace thermofem {
    /** @brief 以连续存储表示一维向量，通过组合管理 CUDA Buffer。
     * @tparam T 满足 BufferElement 的元素类型。
     * @details 向量拥有一个 CudaBuffer，长度直接由存储决定，不重复维护维度状态。
     * 不继承 Buffer，也不提供隐式转换；跨抽象边界时显式使用只读 values()。
     * 复制深拷贝并采用来源策略；移动接管来源存储且来源变为空向量。
     * copyDataFrom 只复制等长数据，保留目标地址及策略；changeMemoryType 可能使指针失效。
     * 所有同步、设备归属和异常保证遵循 CudaBuffer；新分配元素未初始化。
     * 此处仅定义存储语义，数值运算由 LinearAlgebra 层提供。
     * @code
     * thermofem::Vector<double> vector(10, thermofem::GlobalMemory{});
     * vector.changeMemoryType(thermofem::PinnedMemory{});
     * @endcode
     */
    template <BufferElement T>
    class Vector {
    public:
        using ValueType = T; ///< 向量元素类型。
        /** @brief 创建空向量。@param strategy 策略。@pre 无额外约束。@return 无返回值。
         * @code Vector<double> x; @endcode */
        explicit Vector(MemoryStrategy strategy = GlobalMemory{}) noexcept : values_(strategy) {}
        /** @brief 分配未初始化元素。@param count 长度。@param strategy 策略。
         * @pre 字节数不溢出，非空分配需要 CUDA。@return 无返回值。@code Vector<double> x(10); @endcode */
        explicit Vector(std::size_t count, MemoryStrategy strategy = GlobalMemory{}) : values_(count, strategy) {}
        /** @brief 从主机构造向量。@param values 输入。@param strategy 策略。
         * @pre 主机数组有效，非空分配需要 CUDA。@return 无返回值。@code Vector<double> x(input); @endcode */
        explicit Vector(std::span<const T> values, MemoryStrategy strategy = GlobalMemory{}) :
            values_(values, strategy) {}
        /** @brief 深拷贝存储及策略。@param other 来源。@pre 外部访问已同步。@return 无返回值。
         * @code auto y = x; @endcode */
        Vector(const Vector& other) = default;
        /** @brief 接管存储并清空来源。@param other 来源。@pre 外部访问已同步。@return 无返回值。
         * @code auto y = std::move(x); @endcode */
        Vector(Vector&& other) noexcept = default;
        /** @brief 深拷贝并采用来源策略，失败不改变目标。@param other 来源。
         * @pre 外部访问已同步。@return 当前对象。@code y = x; @endcode */
        Vector& operator=(const Vector& other) = default;
        /** @brief 接管来源并释放旧存储。@param other 来源。@pre 外部访问已同步。
         * @return 当前对象。@code y = std::move(x); @endcode */
        Vector& operator=(Vector&& other) noexcept = default;
        /** @brief 自动释放存储，无参数。@pre 外部访问已同步。@return 无返回值。
         * @code { Vector<double> x(3); } @endcode */
        ~Vector() = default;
        /** @brief 复制向量数据并保留目标存储。@param other 来源。
         * @pre 等长、同设备且外部访问已同步。@return 无返回值。@code y.copyDataFrom(x); @endcode */
        void copyDataFrom(const Vector& other) { values_.copyDataFrom(other.values_); }
        /** @brief 显式导入 Buffer 数据并保留目标存储。@param other 来源。
         * @pre 等长、同设备且外部访问已同步。@return 无返回值。@code x.copyDataFrom(buffer); @endcode */
        void copyDataFrom(const CudaBuffer<T>& other) { values_.copyDataFrom(other); }
        /** @brief 保留数据并转换策略，失败不改变对象。@param strategy 策略。
         * @pre 外部访问已同步。@return 无返回值。@code x.changeMemoryType(PinnedMemory{}); @endcode */
        void changeMemoryType(MemoryStrategy strategy) { values_.changeMemoryType(strategy); }
        /** @brief 从主机同步导入。@param values 输入。@pre 等长，外部访问已同步。
         * @return 无返回值。@code x.copyFromHost(input); @endcode */
        void copyFromHost(std::span<const T> values) { values_.copyFromHost(values); }
        /** @brief 同步导出至主机。@param values 输出。@pre 等长，外部访问已同步。
         * @return 无返回值。@code x.copyToHost(output); @endcode */
        void copyToHost(std::span<T> values) const { values_.copyToHost(values); }
        /** @brief 交换所有权与策略。@param other 另一向量。@pre 无额外约束。
         * @return 无返回值。@code x.swap(y); @endcode */
        void swap(Vector& other) noexcept { values_.swap(other.values_); }
        /** @brief 查询长度，无参数、无额外约束。@return 长度。@code auto n = x.size(); @endcode */
        [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }
        /** @brief 查询空状态，无参数、无额外约束。@return 是否为空。@code bool empty = x.empty(); @endcode */
        [[nodiscard]] bool empty() const noexcept { return values_.empty(); }
        /** @brief 查询字节数，无参数、无额外约束。@return 字节数。@code auto bytes = x.sizeBytes(); @endcode */
        [[nodiscard]] std::size_t sizeBytes() const noexcept { return values_.sizeBytes(); }
        /** @brief 查询所属设备，无参数、无额外约束。@return 设备编号，空时为 -1。@code auto d = x.deviceId(); @endcode
         */
        [[nodiscard]] int deviceId() const noexcept { return values_.deviceId(); }
        /** @brief 查询内存类型，无参数、无额外约束。@return 类型。@code auto t = x.memoryType(); @endcode */
        [[nodiscard]] MemoryType memoryType() const noexcept { return values_.memoryType(); }
        /** @brief 查询内存策略，无参数、无额外约束。@return 只读策略。@code auto p = x.memoryStrategy(); @endcode */
        [[nodiscard]] const MemoryStrategy& memoryStrategy() const noexcept { return values_.memoryStrategy(); }
        /** @brief 显式访问只读存储，无参数。@pre 引用仅在向量存活期间有效。
         * @return 内部 Buffer，只读接口禁止外部替换长度及策略。@code auto n = x.values().size(); @endcode */
        [[nodiscard]] const CudaBuffer<T>& values() const noexcept { return values_; }
        /** @brief 获取原始地址，无参数。@pre 按内存策略在正确一端使用。
         * @return 原始地址。@code auto* p = x.data(); @endcode */
        [[nodiscard]] T* data() noexcept { return values_.data(); }
        /** @brief 获取只读原始地址，无参数。@pre 按内存策略使用。
         * @return 只读地址。@code auto* p = std::as_const(x).data(); @endcode */
        [[nodiscard]] const T* data() const noexcept { return values_.data(); }
        /** @brief 获取主机地址，无参数。@pre 非空 global 不可调用，外部访问已同步。
         * @return 主机地址。@code auto* p = x.hostData(); @endcode */
        [[nodiscard]] T* hostData() { return values_.hostData(); }
        /** @brief 获取只读主机地址，无参数。@pre 同 hostData()。
         * @return 只读地址。@code auto* p = std::as_const(x).hostData(); @endcode */
        [[nodiscard]] const T* hostData() const { return values_.hostData(); }
        /** @brief 获取设备地址，无参数。@pre 非空 pinned 不可调用，使用所属设备。
         * @return 设备地址。@code auto* p = x.deviceData(); @endcode */
        [[nodiscard]] T* deviceData() { return values_.deviceData(); }
        /** @brief 获取只读设备地址，无参数。@pre 同 deviceData()。
         * @return 只读地址。@code auto* p = std::as_const(x).deviceData(); @endcode */
        [[nodiscard]] const T* deviceData() const { return values_.deviceData(); }

    private:
        CudaBuffer<T> values_; ///< 独占向量的连续元素存储，不与其他向量共享所有权。
    };
} // namespace thermofem

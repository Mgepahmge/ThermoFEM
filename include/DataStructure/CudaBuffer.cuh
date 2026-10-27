/** @file CudaBuffer.cuh
 * @brief 定义具有值语义和可变内存策略的类型化 CUDA 缓冲区。
 */
#pragma once

#include <DataStructure/Memory.cuh>

#include <span>
#include <type_traits>
#include <utility>

namespace thermofem {
    /// 限定可按字节传输且不需要运行构造、析构函数的非 cv 元素。
    template <class T>
    concept BufferElement = std::is_trivially_copyable_v<T> && std::is_trivially_default_constructible_v<T> &&
        std::is_trivially_destructible_v<T> && !std::is_const_v<T> && !std::is_volatile_v<T> && !std::is_pointer_v<T> &&
        (alignof(T) <= alignof(std::max_align_t));

    /** @brief 管理连续 CUDA 存储，深拷贝和移动均采用来源内存策略。
     * @tparam T 满足 BufferElement 的元素类型，元素初值未定义。
     * @details 同步复制接口要求调用方先完成外部异步读写。数据指针遵循当前内存类型的
     * 可访问性，传入内核时使用 deviceData()；类型转换会使已有指针失效。
     * @code
     * thermofem::CudaBuffer<double> values(3, thermofem::PinnedMemory{});
     * values.hostData()[0] = 1.0;
     * values.changeMemoryType(thermofem::GlobalMemory{});
     * @endcode
     */
    template <BufferElement T>
    class CudaBuffer {
    public:
        /** @brief 创建空缓冲区。@param strategy 内存策略。@pre 无额外约束。@return 无返回值。
         * @code thermofem::CudaBuffer<double> buffer; @endcode
         */
        explicit CudaBuffer(MemoryStrategy strategy = GlobalMemory{}) noexcept : storage_(strategy) {}
        /** @brief 分配指定数量的未初始化元素。@param count 元素数。@param strategy 内存策略。
         * @pre 字节数不溢出，非零分配需要 CUDA。@return 无返回值。
         * @code thermofem::CudaBuffer<double> buffer(10); @endcode
         */
        explicit CudaBuffer(std::size_t count, MemoryStrategy strategy = GlobalMemory{}) :
            storage_(detail::checkedProduct(count, sizeof(T)), strategy) {}
        /** @brief 从主机数组构造缓冲区。@param values 主机元素。@param strategy 内存策略。
         * @pre values 有效，非零分配需要 CUDA。@return 无返回值。
         * @code thermofem::CudaBuffer<double> buffer(std::span<const double>(values)); @endcode
         */
        explicit CudaBuffer(std::span<const T> values, MemoryStrategy strategy = GlobalMemory{}) :
            CudaBuffer(values.size(), strategy) {
            copyFromHost(values);
        }
        /** @brief 深拷贝数据及策略。@param other 来源。@pre 异步访问已结束。@return 无返回值。
         * @code auto copy = buffer; @endcode
         */
        CudaBuffer(const CudaBuffer& other) = default;
        /** @brief 接管数据及策略。@param other 来源，随后为空。@pre 无额外约束。@return 无返回值。
         * @code auto moved = std::move(buffer); @endcode
         */
        CudaBuffer(CudaBuffer&& other) noexcept = default;
        /** @brief 深拷贝并覆盖原内存类型。@param other 来源。@pre 异步访问已结束。
         * @return 当前对象。@code target = source; @endcode
         */
        CudaBuffer& operator=(const CudaBuffer& other) = default;
        /** @brief 移动并覆盖原内存类型。@param other 来源，随后为空。@pre 异步访问已结束。
         * @return 当前对象。@code target = std::move(source); @endcode
         */
        CudaBuffer& operator=(CudaBuffer&& other) noexcept = default;
        /** @brief 自动释放资源，无参数。@pre 外部异步访问已结束。@return 无返回值。
         * @code { thermofem::CudaBuffer<double> buffer; } @endcode
         */
        ~CudaBuffer() = default;
        /** @brief 同步复制全部数据而保留目标类型及地址。@param other 来源。
         * @pre 元素数相同且非空存储属于同一设备，异步访问已结束。@return 无返回值。
         * @code target.copyDataFrom(source); @endcode
         */
        void copyDataFrom(const CudaBuffer& other) { storage_.copyDataFrom(other.storage_); }
        /** @brief 保留数据并转换内存类型，失败不改变对象。@param strategy 新策略。
         * @pre 异步访问已结束。@return 无返回值。
         * @code buffer.changeMemoryType(thermofem::MappedMemory{}); @endcode
         */
        void changeMemoryType(MemoryStrategy strategy) { storage_.changeMemoryType(strategy); }
        /** @brief 从主机同步导入全部元素。@param values 主机数组。
         * @pre 元素数相同，异步访问已结束。@return 无返回值。
         * @code buffer.copyFromHost(values); @endcode
         */
        void copyFromHost(std::span<const T> values) {
            storage_.copyFromHost(values.data(), detail::checkedProduct(values.size(), sizeof(T)));
        }
        /** @brief 同步导出全部元素至主机。@param values 主机数组。
         * @pre 元素数相同，异步访问已结束。@return 无返回值。
         * @code buffer.copyToHost(values); @endcode
         */
        void copyToHost(std::span<T> values) const {
            storage_.copyToHost(values.data(), detail::checkedProduct(values.size(), sizeof(T)));
        }
        /** @brief 交换所有权及策略。@param other 交换对象。@pre 无额外约束。@return 无返回值。
         * @code target.swap(source); @endcode
         */
        void swap(CudaBuffer& other) noexcept { storage_.swap(other.storage_); }
        /** @brief 查询元素数，无参数、无额外约束。@return 元素数。
         * @code const auto count = buffer.size(); @endcode
         */
        [[nodiscard]] std::size_t size() const noexcept { return storage_.sizeBytes() / sizeof(T); }
        /** @brief 查询是否为空，无参数、无额外约束。@return 空时为真。
         * @code const bool isEmpty = buffer.empty(); @endcode
         */
        [[nodiscard]] bool empty() const noexcept { return size() == 0; }
        /** @brief 查询字节数，无参数、无额外约束。@return 字节数。
         * @code const auto bytes = buffer.sizeBytes(); @endcode
         */
        [[nodiscard]] std::size_t sizeBytes() const noexcept { return storage_.sizeBytes(); }
        /** @brief 查询分配所属设备，无参数、无额外约束。@return 设备编号，空缓冲区为 -1。
         * @code const int device = buffer.deviceId(); @endcode
         */
        [[nodiscard]] int deviceId() const noexcept { return storage_.deviceId(); }
        /** @brief 查询类型，无参数、无额外约束。@return 内存类别。
         * @code const auto type = buffer.memoryType(); @endcode
         */
        [[nodiscard]] MemoryType memoryType() const noexcept { return storage_.memoryType(); }
        /** @brief 查询策略，无参数、无额外约束。@return 策略常量引用。
         * @code const auto& strategy = buffer.memoryStrategy(); @endcode
         */
        [[nodiscard]] const MemoryStrategy& memoryStrategy() const noexcept { return storage_.memoryStrategy(); }
        /** @brief 获取原始指针，无参数。@pre 按内存类型在合适一端使用。@return 原始地址。
         * @code auto* pointer = buffer.data(); @endcode
         */
        [[nodiscard]] T* data() noexcept { return static_cast<T*>(storage_.data()); }
        /** @brief 获取只读原始指针，无参数。@pre 按类型使用。@return 只读原始地址。
         * @code const auto* pointer = std::as_const(buffer).data(); @endcode
         */
        [[nodiscard]] const T* data() const noexcept { return static_cast<const T*>(storage_.data()); }
        /** @brief 获取主机地址，无参数。@pre 非空 global 不可调用，异步访问已结束。
         * @return 主机地址。@code auto* pointer = buffer.hostData(); @endcode
         */
        [[nodiscard]] T* hostData() { return static_cast<T*>(storage_.hostData()); }
        /** @brief 获取只读主机地址，无参数。@pre 同 hostData()。@return 只读主机地址。
         * @code const auto* pointer = std::as_const(buffer).hostData(); @endcode
         */
        [[nodiscard]] const T* hostData() const { return static_cast<const T*>(storage_.hostData()); }
        /** @brief 获取设备地址，无参数。@pre 非空 pinned 不可调用，内核使用分配所属设备。
         * @return 设备地址。@code auto* pointer = buffer.deviceData(); @endcode
         */
        [[nodiscard]] T* deviceData() { return static_cast<T*>(storage_.deviceData()); }
        /** @brief 获取只读设备地址，无参数。@pre 同 deviceData()。@return 只读设备地址。
         * @code const auto* pointer = std::as_const(buffer).deviceData(); @endcode
         */
        [[nodiscard]] const T* deviceData() const { return static_cast<const T*>(storage_.deviceData()); }

    private:
        detail::MemoryStorage storage_; ///< 负责字节存储、设备归属和策略的 RAII 对象。
    };
} // namespace thermofem

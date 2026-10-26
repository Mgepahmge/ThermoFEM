/** @file Memory.cuh
 * @brief 定义 CUDA 内存策略、错误类型与按字节管理的 RAII 存储。
 */
#pragma once

#include <cuda_runtime_api.h>

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <variant>

namespace thermofem {

/// CUDA 内存类别，分别表示设备、页锁定主机、映射主机及统一内存。
enum class MemoryType { Global, Pinned, Mapped, Managed };

/** @brief 使用 cudaMalloc 分配设备全局内存的策略。
 * @code thermofem::GlobalMemory policy; @endcode
 */
struct GlobalMemory {};
/** @brief 使用 cudaMallocHost 分配页锁定主机内存的策略。
 * @code thermofem::PinnedMemory policy; @endcode
 */
struct PinnedMemory {};
/** @brief 使用 cudaHostAllocMapped 分配设备可映射主机内存的策略。
 * @code thermofem::MappedMemory policy; @endcode
 */
struct MappedMemory {};
/** @brief 使用 cudaMallocManaged 分配统一内存的策略。
 * @code thermofem::ManagedMemory policy; @endcode
 */
struct ManagedMemory {};

/// 限定为项目支持的 CUDA 分配策略。
template<class T>
concept MemoryPolicy = std::same_as<T, GlobalMemory> || std::same_as<T, PinnedMemory>
    || std::same_as<T, MappedMemory> || std::same_as<T, ManagedMemory>;

/// 保存运行时可切换的内存策略，通过静态重载分派实际分配操作。
using MemoryStrategy = std::variant<GlobalMemory, PinnedMemory, MappedMemory, ManagedMemory>;

/** @brief 保存 CUDA 运行时失败信息的异常。
 * @code
 * try { throw thermofem::CudaError(cudaErrorMemoryAllocation, "cudaMalloc"); }
 * catch (const thermofem::CudaError& error) {
 *     const auto code = error.code();
 * }
 * @endcode
 */
class CudaError : public std::runtime_error {
public:
    /** @brief 构造 CUDA 异常。@param code CUDA 错误码。@param operation 操作名称。
     * @pre operation 非空。@return 无返回值。
     * @code thermofem::CudaError error(cudaErrorMemoryAllocation, "cudaMalloc"); @endcode
     */
    CudaError(cudaError_t code, const char* operation);
    /** @brief 查询错误码，无参数、无额外约束。@return CUDA 错误码。
     * @code const auto code = error.code(); @endcode
     */
    [[nodiscard]] cudaError_t code() const noexcept;
private:
    cudaError_t code_; ///< 触发异常的 CUDA 错误码。
};

/** @brief 提供数据结构内部使用的存储与尺寸检查设施。 */
namespace detail {

/** @brief 检查字节数或维度乘积。@param left 左操作数。@param right 右操作数。
 * @pre 乘积可由 size_t 表示，否则抛出 length_error。@return 无溢出的乘积。
 * @code const auto bytes = thermofem::detail::checkedProduct(10, sizeof(double)); @endcode
 */
[[nodiscard]] std::size_t checkedProduct(std::size_t left, std::size_t right);

/** @brief 按字节管理 CUDA 内存，在复制、移动和转换时维护分配策略及设备归属。
 * @details 非空分配绑定创建时的设备；复制在同一设备完成，不支持跨设备复制。
 * 析构不抛异常，尽力释放资源；调用方须在释放或主机访问前完成外部异步操作。
 * @code thermofem::detail::MemoryStorage storage(64, thermofem::PinnedMemory{}); @endcode
 */
class MemoryStorage {
public:
    /** @brief 创建空存储。@param strategy 内存策略。@pre 无额外约束。@return 无返回值。
     * @code thermofem::detail::MemoryStorage storage; @endcode
     */
    explicit MemoryStorage(MemoryStrategy strategy = GlobalMemory{}) noexcept;
    /** @brief 分配未初始化存储。@param bytes 字节数。@param strategy 内存策略。
     * @pre 非零分配需要可用的 CUDA 设备。@return 无返回值。
     * @code thermofem::detail::MemoryStorage storage(64, thermofem::GlobalMemory{}); @endcode
     */
    MemoryStorage(std::size_t bytes, MemoryStrategy strategy);
    /** @brief 深拷贝来源数据及策略。@param other 来源。@pre 来源没有未完成的异步访问。
     * @return 无返回值。@code auto copy = storage; @endcode
     */
    MemoryStorage(const MemoryStorage& other);
    /** @brief 接管来源所有权及策略。@param other 来源，随后为空。
     * @pre 无额外约束。@return 无返回值。@code auto moved = std::move(storage); @endcode
     */
    MemoryStorage(MemoryStorage&& other) noexcept;
    /** @brief 以强异常保证深拷贝并覆盖策略。@param other 来源。@pre 外部异步操作已结束。
     * @return 当前对象。@code target = source; @endcode
     */
    MemoryStorage& operator=(const MemoryStorage& other);
    /** @brief 释放原存储并接管来源及策略。@param other 来源，随后为空。
     * @pre 外部异步操作已结束。@return 当前对象。@code target = std::move(source); @endcode
     */
    MemoryStorage& operator=(MemoryStorage&& other) noexcept;
    /** @brief 释放资源，无参数。@pre 外部异步操作已结束。@return 无返回值。
     * @code { thermofem::detail::MemoryStorage storage; } @endcode
     */
    ~MemoryStorage() noexcept;
    /** @brief 交换存储元数据与所有权。@param other 交换对象。@pre 无额外约束。
     * @return 无返回值。@code target.swap(source); @endcode
     */
    void swap(MemoryStorage& other) noexcept;
    /** @brief 同步复制数据，保留目标策略和地址。@param other 来源。
     * @pre 字节数相同，非空存储属于同一设备，外部异步操作已结束。
     * @return 无返回值。@code target.copyDataFrom(source); @endcode
     */
    void copyDataFrom(const MemoryStorage& other);
    /** @brief 保留数据并切换策略，失败时保持原对象。@param strategy 新策略。
     * @pre 外部异步操作已结束。@return 无返回值。
     * @code storage.changeMemoryType(thermofem::PinnedMemory{}); @endcode
     */
    void changeMemoryType(MemoryStrategy strategy);
    /** @brief 从主机同步导入全部字节。@param source 主机地址。@param bytes 字节数。
     * @pre 尺寸匹配，非空源有效且与目标不重叠，异步访问已结束。@return 无返回值。
     * @code storage.copyFromHost(values, sizeof(values)); @endcode
     */
    void copyFromHost(const void* source, std::size_t bytes);
    /** @brief 同步导出全部字节至主机。@param target 主机地址。@param bytes 字节数。
     * @pre 尺寸匹配，非空目标有效且与源不重叠，异步访问已结束。@return 无返回值。
     * @code storage.copyToHost(values, sizeof(values)); @endcode
     */
    void copyToHost(void* target, std::size_t bytes) const;
    /** @brief 获取分配地址，无参数、无额外约束。@return 原始地址，空存储返回空指针。
     * @code auto* pointer = storage.data(); @endcode
     */
    [[nodiscard]] void* data() noexcept;
    /** @brief 获取只读分配地址，无参数、无额外约束。@return 只读原始地址。
     * @code const auto* pointer = std::as_const(storage).data(); @endcode
     */
    [[nodiscard]] const void* data() const noexcept;
    /** @brief 获取设备可访问地址，无参数。@pre 非空 pinned 存储不可调用。
     * @return 设备地址，mapped 存储返回设备别名，空存储返回空指针。
     * @code auto* pointer = storage.deviceData(); @endcode
     */
    [[nodiscard]] void* deviceData() const;
    /** @brief 获取主机可访问地址，无参数。@pre 非空 global 存储不可调用。
     * @return 主机地址，空存储返回空指针。@code auto* pointer = storage.hostData(); @endcode
     */
    [[nodiscard]] void* hostData() const;
    /** @brief 查询字节数，无参数、无额外约束。@return 字节数。
     * @code const auto bytes = storage.sizeBytes(); @endcode
     */
    [[nodiscard]] std::size_t sizeBytes() const noexcept;
    /** @brief 查询所属设备，无参数、无额外约束。@return 设备编号，空存储为 -1。
     * @code const int device = storage.deviceId(); @endcode
     */
    [[nodiscard]] int deviceId() const noexcept;
    /** @brief 查询内存类型，无参数、无额外约束。@return 内存类别。
     * @code const auto type = storage.memoryType(); @endcode
     */
    [[nodiscard]] MemoryType memoryType() const noexcept;
    /** @brief 查询策略，无参数、无额外约束。@return 策略对象的常量引用。
     * @code const auto& strategy = storage.memoryStrategy(); @endcode
     */
    [[nodiscard]] const MemoryStrategy& memoryStrategy() const noexcept;
private:
    void release() noexcept; ///< 在所属设备上释放资源，并清空存储状态。
    void* data_{}; ///< CUDA 分配返回的原始地址。
    std::size_t bytes_{}; ///< 当前分配的字节数。
    int device_{-1}; ///< 分配所属 CUDA 设备，空存储为 -1。
    MemoryStrategy strategy_; ///< 控制分配与释放方式的策略对象。
};
} // 命名空间 detail
} // 命名空间 thermofem

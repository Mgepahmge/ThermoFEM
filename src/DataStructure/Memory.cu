/** @file Memory.cu
 * @brief 实现 CUDA 内存策略分派、设备归属管理与同步数据传输。
 */
#include <DataStructure/Memory.cuh>

#include <cstring>
#include <limits>
#include <utility>

namespace thermofem {
    namespace detail {
        namespace {
            /** @brief 检查 CUDA 调用结果，失败时抛出异常。
             * @param result CUDA 结果。@param operation 操作名称。@pre operation 有效。@note 无返回值。
             * @code checkCuda(cudaGetLastError(), "kernel launch"); @endcode
             */
            void checkCuda(cudaError_t result, const char* operation) {
                if (result != cudaSuccess) {
                    throw CudaError(result, operation);
                }
            }

            /** @brief 在操作期间激活所属设备，退出时恢复调用方设备。
             * @code DeviceScope scope(device); @endcode
             */
            class DeviceScope {
            public:
                /** @brief 激活指定设备。@param device 目标设备。@pre 编号有效。@note 无返回值。
                 * @code DeviceScope scope(0); @endcode
                 */
                explicit DeviceScope(int device) {
                    checkCuda(cudaGetDevice(&previousDevice_), "cudaGetDevice");
                    if (previousDevice_ != device) {
                        checkCuda(cudaSetDevice(device), "cudaSetDevice");
                    }
                }
                /** @brief 尽力恢复原设备，无参数、无额外约束。@note 无返回值。
                 * @code { DeviceScope scope(0); } @endcode
                 */
                ~DeviceScope() noexcept { (void)cudaSetDevice(previousDevice_); }
                /// 禁止复制设备切换守卫，防止重复恢复上下文。
                DeviceScope(const DeviceScope&) = delete;
                /// 禁止复制赋值，防止丢失原设备状态。
                DeviceScope& operator=(const DeviceScope&) = delete;

            private:
                int previousDevice_{}; ///< 调用方原先激活的设备编号。
            };

            /** @brief 分配全局内存。@param bytes 字节数。@pre bytes 非零且设备可用。
             * @return 分配地址，失败抛异常。@code auto* pointer = allocate(GlobalMemory{}, 64); @endcode
             */
            void* allocate(GlobalMemory, std::size_t bytes) {
                void* result{};
                checkCuda(cudaMalloc(&result, bytes), "cudaMalloc");
                return result;
            }
            /** @brief 分配页锁定主机内存。@param bytes 字节数。@pre bytes 非零且 CUDA 可用。
             * @return 分配地址，失败抛异常。@code auto* pointer = allocate(PinnedMemory{}, 64); @endcode
             */
            void* allocate(PinnedMemory, std::size_t bytes) {
                void* result{};
                checkCuda(cudaMallocHost(&result, bytes), "cudaMallocHost");
                return result;
            }
            /** @brief 分配映射主机内存。@param bytes 字节数。@pre bytes 非零且设备支持映射。
             * @return 主机地址，失败抛异常。@code auto* pointer = allocate(MappedMemory{}, 64); @endcode
             */
            void* allocate(MappedMemory, std::size_t bytes) {
                int device{};
                int supported{};
                checkCuda(cudaGetDevice(&device), "cudaGetDevice");
                checkCuda(cudaDeviceGetAttribute(&supported, cudaDevAttrCanMapHostMemory, device),
                          "cudaDeviceGetAttribute");
                if (!supported) {
                    throw std::runtime_error("CUDA device cannot map host memory");
                }
                void* result{};
                checkCuda(cudaHostAlloc(&result, bytes, cudaHostAllocMapped), "cudaHostAlloc");
                return result;
            }
            /** @brief 分配统一内存。@param bytes 字节数。@pre bytes 非零且设备支持统一内存。
             * @return 分配地址，失败抛异常。@code auto* pointer = allocate(ManagedMemory{}, 64); @endcode
             */
            void* allocate(ManagedMemory, std::size_t bytes) {
                void* result{};
                checkCuda(cudaMallocManaged(&result, bytes), "cudaMallocManaged");
                return result;
            }
            /** @brief 尽力释放全局内存。@param pointer 所属设备上的分配地址。
             * @pre 外部异步访问已结束。@note 无返回值。@code deallocate(GlobalMemory{}, pointer); @endcode
             */
            void deallocate(GlobalMemory, void* pointer) noexcept { (void)cudaFree(pointer); }
            /** @brief 尽力释放页锁定主机内存。@param pointer 对应分配地址。
             * @pre 外部异步访问已结束。@note 无返回值。@code deallocate(PinnedMemory{}, pointer); @endcode
             */
            void deallocate(PinnedMemory, void* pointer) noexcept { (void)cudaFreeHost(pointer); }
            /** @brief 尽力释放映射主机内存。@param pointer 主机分配地址。
             * @pre 外部异步访问已结束。@note 无返回值。@code deallocate(MappedMemory{}, pointer); @endcode
             */
            void deallocate(MappedMemory, void* pointer) noexcept { (void)cudaFreeHost(pointer); }
            /** @brief 尽力释放统一内存。@param pointer 所属设备上的分配地址。
             * @pre 外部异步访问已结束。@note 无返回值。@code deallocate(ManagedMemory{}, pointer); @endcode
             */
            void deallocate(ManagedMemory, void* pointer) noexcept { (void)cudaFree(pointer); }
        } // namespace

        std::size_t checkedProduct(std::size_t left, std::size_t right) {
            if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
                throw std::length_error("Data structure size overflows size_t");
            }
            return left * right;
        }

        MemoryStorage::MemoryStorage(MemoryStrategy strategy) noexcept : strategy_(strategy) {}

        MemoryStorage::MemoryStorage(std::size_t bytes, MemoryStrategy strategy) : strategy_(strategy) {
            if (bytes == 0) {
                return;
            }
            checkCuda(cudaGetDevice(&device_), "cudaGetDevice");
            data_ =
                std::visit([bytes]<MemoryPolicy Policy>(Policy policy) { return allocate(policy, bytes); }, strategy_);
            bytes_ = bytes;
        }

        MemoryStorage::MemoryStorage(const MemoryStorage& other) : strategy_(other.strategy_) {
            if (other.bytes_ != 0) {
                DeviceScope deviceScope(other.device_);
                MemoryStorage temporary(other.bytes_, other.strategy_);
                temporary.copyDataFrom(other);
                swap(temporary);
            }
        }

        MemoryStorage::MemoryStorage(MemoryStorage&& other) noexcept : strategy_(other.strategy_) { swap(other); }

        MemoryStorage& MemoryStorage::operator=(const MemoryStorage& other) {
            if (this != &other) {
                MemoryStorage temporary(other);
                swap(temporary);
            }
            return *this;
        }

        MemoryStorage& MemoryStorage::operator=(MemoryStorage&& other) noexcept {
            if (this != &other) {
                MemoryStorage temporary(std::move(other));
                swap(temporary);
            }
            return *this;
        }

        MemoryStorage::~MemoryStorage() noexcept { release(); }

        void MemoryStorage::release() noexcept {
            if (data_ != nullptr) {
                int previousDevice{};
                const bool canRestore = cudaGetDevice(&previousDevice) == cudaSuccess;
                (void)cudaSetDevice(device_);
                std::visit([this]<MemoryPolicy Policy>(Policy policy) { deallocate(policy, data_); }, strategy_);
                if (canRestore) {
                    (void)cudaSetDevice(previousDevice);
                }
            }
            data_ = nullptr;
            bytes_ = 0;
            device_ = -1;
        }

        void MemoryStorage::swap(MemoryStorage& other) noexcept {
            using std::swap;
            swap(data_, other.data_);
            swap(bytes_, other.bytes_);
            swap(device_, other.device_);
            swap(strategy_, other.strategy_);
        }

        void MemoryStorage::copyDataFrom(const MemoryStorage& other) {
            if (bytes_ != other.bytes_) {
                throw std::invalid_argument("Buffer sizes must match");
            }
            if (bytes_ == 0 || this == &other) {
                return;
            }
            if (device_ != other.device_) {
                throw std::invalid_argument("Cross-device copies are not supported");
            }
            DeviceScope deviceScope(device_);
            if (memoryType() != MemoryType::Global && other.memoryType() != MemoryType::Global) {
                std::memmove(data_, other.data_, bytes_);
            }
            else {
                checkCuda(cudaMemcpy(data_, other.data_, bytes_, cudaMemcpyDefault), "cudaMemcpy");
            }
        }

        void MemoryStorage::changeMemoryType(MemoryStrategy strategy) {
            if (strategy_.index() == strategy.index()) {
                return;
            }
            if (bytes_ == 0) {
                strategy_ = strategy;
                return;
            }
            DeviceScope deviceScope(device_);
            MemoryStorage temporary(bytes_, strategy);
            temporary.copyDataFrom(*this);
            swap(temporary);
        }

        void MemoryStorage::copyFromHost(const void* source, std::size_t bytes) {
            if (bytes != bytes_ || (bytes != 0 && source == nullptr)) {
                throw std::invalid_argument("Invalid host source or size");
            }
            if (bytes == 0) {
                return;
            }
            DeviceScope deviceScope(device_);
            if (memoryType() == MemoryType::Global) {
                checkCuda(cudaMemcpy(data_, source, bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
            }
            else {
                std::memmove(data_, source, bytes);
            }
        }

        void MemoryStorage::copyToHost(void* target, std::size_t bytes) const {
            if (bytes != bytes_ || (bytes != 0 && target == nullptr)) {
                throw std::invalid_argument("Invalid host target or size");
            }
            if (bytes == 0) {
                return;
            }
            DeviceScope deviceScope(device_);
            if (memoryType() == MemoryType::Global) {
                checkCuda(cudaMemcpy(target, data_, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy");
            }
            else {
                std::memmove(target, data_, bytes);
            }
        }

        void* MemoryStorage::data() noexcept { return data_; }
        const void* MemoryStorage::data() const noexcept { return data_; }
        void* MemoryStorage::deviceData() const {
            if (bytes_ == 0) {
                return nullptr;
            }
            if (memoryType() == MemoryType::Pinned) {
                throw std::logic_error("Pinned memory has no guaranteed device view; use MappedMemory");
            }
            if (memoryType() == MemoryType::Mapped) {
                DeviceScope deviceScope(device_);
                void* result{};
                checkCuda(cudaHostGetDevicePointer(&result, data_, 0), "cudaHostGetDevicePointer");
                return result;
            }
            return data_;
        }
        void* MemoryStorage::hostData() const {
            if (bytes_ != 0 && memoryType() == MemoryType::Global) {
                throw std::logic_error("Global memory is not host accessible");
            }
            return data_;
        }
        std::size_t MemoryStorage::sizeBytes() const noexcept { return bytes_; }
        int MemoryStorage::deviceId() const noexcept { return device_; }
        MemoryType MemoryStorage::memoryType() const noexcept { return static_cast<MemoryType>(strategy_.index()); }
        const MemoryStrategy& MemoryStorage::memoryStrategy() const noexcept { return strategy_; }
    } // namespace detail

    CudaError::CudaError(cudaError_t code, const char* operation) :
        std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code)), code_(code) {}
    cudaError_t CudaError::code() const noexcept { return code_; }
} // namespace thermofem

/** @file example-2.cu
 * @brief 演示向量、稠密矩阵、CSR 矩阵以及不同内存策略上的内核访问。
 */
#include <DataStructure/CsrMatrix.cuh>
#include <DataStructure/DenseMatrix.cuh>
#include <DataStructure/Vector.cuh>

#include <array>
#include <iostream>

/** @brief 包含数据结构示例的内核与错误检查辅助函数。 */
namespace {
    /** @brief 在 GPU 上按指定倍数缩放连续元素。
     * @param values 可写设备指针。@param count 元素数。@param scale 缩放倍数。
     * @pre values 覆盖 count 个元素，所属设备是当前设备。@return 无返回值。
     * @code scaleValues<<<1, 32>>>(vector.deviceData(), vector.size(), 2.0); @endcode
     */
    __global__ void scaleValues(double* values, std::size_t count, double scale) {
        const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index < count) {
            values[index] *= scale;
        }
    }

    /** @brief 将 CUDA 失败转换为项目异常。@param result CUDA 结果。@param operation 操作名称。
     * @pre operation 有效。@return 无返回值。
     * @code checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); @endcode
     */
    void checkCuda(cudaError_t result, const char* operation) {
        if (result != cudaSuccess) {
            throw thermofem::CudaError(result, operation);
        }
    }
} // namespace

/** @brief 在三种设备可访问内存中运行向量内核，并验证稠密与稀疏矩阵存储。
 * @details 无参数，输出各策略的内核验证结果。
 * @pre 设备支持映射内存和统一内存。@return 成功返回 0，发生异常时返回 1。
 * @code ./build/example/example-2 @endcode
 */
int main() {
    using namespace thermofem;
    try {
        const std::array<double, 4> input{1, 2, 3, 4};
        const std::array<double, 4> expected{2, 4, 6, 8};
        const std::array<MemoryStrategy, 3> strategies{GlobalMemory{}, MappedMemory{}, ManagedMemory{}};
        for (const auto& strategy : strategies) {
            Vector<double> vector(input, strategy);
            scaleValues<<<1, 32>>>(vector.deviceData(), vector.size(), 2.0);
            checkCuda(cudaGetLastError(), "scaleValues launch");
            checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            std::array<double, 4> result{};
            vector.copyToHost(result);
            if (result != expected) {
                throw std::runtime_error("Kernel result mismatch");
            }
            std::cout << "Kernel access verified for memory type " << static_cast<int>(vector.memoryType()) << '\n';
        }
        DenseMatrix<double> dense(2, 2, PinnedMemory{});
        dense.copyFromHost(input);
        if (dense.at(1, 0) != 3) {
            throw std::runtime_error("Dense layout mismatch");
        }
        const std::array<int, 3> offsets{0, 1, 2};
        const std::array<int, 2> columns{0, 1};
        const std::array<double, 2> entries{2, 3};
        CsrMatrix<double> sparse(2, 2, offsets, columns, entries, PinnedMemory{});
        sparse.changeMemoryType(GlobalMemory{});
        std::array<double, 2> result{};
        sparse.values().copyToHost(result);
        if (result != entries) {
            throw std::runtime_error("CSR conversion mismatch");
        }
        std::cout << "Dense and CSR matrix storage verified\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

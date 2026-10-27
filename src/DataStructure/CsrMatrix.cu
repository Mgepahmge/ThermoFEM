/** @file CsrMatrix.cu
 * @brief 验证 GPU 生成的 CSR 缓冲区，避免全量主机往返。
 */
#include <DataStructure/CsrMatrix.cuh>
#include <LinearAlgebra/Operations.cuh>
namespace thermofem::detail {
    /** @brief 提供 CSR 结构校验内核。 */
    namespace {
        /** @brief 检查行区间及其列索引。@param rows 行数。@param columns 列数。@param count 非零数。
         * @param offsets 行偏移。@param indices 列索引。@param bad 错误标志。
         * @pre 当前设备、有效数组。@note 无返回值。@code validate<<<1,256>>>(n,n,nz,rp,ci,bad); @endcode */
        template <CsrIndex I>
        __global__ void validate(std::size_t rows, std::size_t columns, std::size_t count, const I* offsets,
                                 const I* indices, int* bad) {
            const std::size_t row = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (row == 0 && (offsets[0] != 0 || offsets[rows] != static_cast<I>(count))) {
                atomicExch(bad, 1);
            }
            if (row >= rows) {
                return;
            }
            if (offsets[row] < 0 || offsets[row + 1] < offsets[row] || offsets[row + 1] > static_cast<I>(count)) {
                atomicExch(bad, 1);
                return;
            }
            for (I j = offsets[row]; j < offsets[row + 1]; ++j) {
                if (indices[j] < 0 || static_cast<std::size_t>(indices[j]) >= columns) {
                    atomicExch(bad, 1);
                }
            }
        }
    } // namespace
    void validateDeviceCsr(std::size_t rows, std::size_t columns, std::size_t count, const void* offsets,
                           const void* indices, bool wide, int device) {
        linalg::detail::OperationScope scope(device);
        CudaBuffer<int> bad(1);
        auto check = [](cudaError_t status) {
            if (status != cudaSuccess) {
                throw CudaError(status, "CSR validation");
            }
        };
        check(cudaMemset(bad.data(), 0, sizeof(int)));
        const unsigned blocks = static_cast<unsigned>(std::max<std::size_t>(1, (rows + 255) / 256));
        if (wide) {
            validate<<<blocks, 256>>>(rows, columns, count, static_cast<const std::int64_t*>(offsets),
                                      static_cast<const std::int64_t*>(indices), bad.data());
        }
        else {
            validate<<<blocks, 256>>>(rows, columns, count, static_cast<const std::int32_t*>(offsets),
                                      static_cast<const std::int32_t*>(indices), bad.data());
        }
        check(cudaGetLastError());
        int result{};
        check(cudaMemcpy(&result, bad.data(), sizeof(int), cudaMemcpyDeviceToHost));
        if (result) {
            throw std::invalid_argument("Invalid device CSR structure");
        }
    }
} // namespace thermofem::detail

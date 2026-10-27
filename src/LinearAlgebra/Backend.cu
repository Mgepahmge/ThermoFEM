/** @file Backend.cu
 * @brief 实现同步 cuBLAS/cuSPARSE 调用及其句柄、描述符的 RAII 管理。
 */
#include <LinearAlgebra/Backend.cuh>
#include <cublas_v2.h>
#include <cusparse.h>
#include <string>
#include <type_traits>

namespace thermofem::linalg::detail {
    /** @brief 包含库错误检查、描述符及精度分派等内部实现。 */
    namespace {
        /** @brief 检查 CUDA 结果。@param status 状态。@param operation 操作。@pre 名称有效。
         * @note 无返回值。@code check(cudaDeviceSynchronize(), "synchronize"); @endcode
         */
        void check(cudaError_t status, const char* operation) {
            if (status != cudaSuccess) {
                throw CudaError(status, operation);
            }
        }
        /** @brief 检查 cuBLAS 结果。@param status 状态。@param operation 操作。@pre 名称有效。
         * @note 无返回值。@code check(cublasCreate(&handle), "cublasCreate"); @endcode
         */
        void check(cublasStatus_t status, const char* operation) {
            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error(std::string(operation) + ": cuBLAS status " + std::to_string(status));
            }
        }
        /** @brief 检查 cuSPARSE 结果。@param status 状态。@param operation 操作。@pre 名称有效。
         * @note 无返回值。@code check(cusparseCreate(&handle), "cusparseCreate"); @endcode
         */
        void check(cusparseStatus_t status, const char* operation) {
            if (status != CUSPARSE_STATUS_SUCCESS) {
                throw std::runtime_error(std::string(operation) + ": " + cusparseGetErrorString(status));
            }
        }
        /** @brief 自动释放 CUDA 库句柄或描述符，不允许复制。
         * @tparam T 句柄指针类型。@tparam Destroy 对应销毁函数。
         * @code Owner<cublasHandle_t, cublasDestroy> handle; @endcode
         */
        template <class T, auto Destroy>
            requires std::is_pointer_v<T> && std::invocable<decltype(Destroy), T>
        class Owner {
        public:
            /** @brief 创建空所有者，无参数、无额外约束。@note 无返回值。
             * @code Owner<cublasHandle_t, cublasDestroy> owner; @endcode
             */
            Owner() = default;
            /** @brief 尽力销毁资源，无参数。@pre 外部调用已经同步。@note 无返回值。
             * @code { Owner<cublasHandle_t, cublasDestroy> owner; } @endcode
             */
            ~Owner() noexcept {
                if (value_) {
                    (void)Destroy(value_);
                }
            }
            Owner(const Owner&) = delete; ///< 禁止复制所有权。
            Owner& operator=(const Owner&) = delete; ///< 禁止复制赋值。
            /** @brief 获取初始化地址，无参数。@pre 仅用于空所有者。@return 句柄指针。
             * @code check(cublasCreate(owner.address()), "create"); @endcode
             */
            T* address() noexcept { return &value_; }
            /** @brief 获取句柄，无参数、无额外约束。@return 句柄值。
             * @code auto handle = owner.get(); @endcode
             */
            T get() const noexcept { return value_; }

        private:
            T value_{}; ///< 唯一持有的句柄或描述符。
        };
        using BlasHandle = Owner<cublasHandle_t, cublasDestroy>;
        using SparseHandle = Owner<cusparseHandle_t, cusparseDestroy>;
        using SparseDescriptor = Owner<cusparseSpMatDescr_t, cusparseDestroySpMat>;
        using VectorDescriptor = Owner<cusparseDnVecDescr_t, cusparseDestroyDnVec>;
        using DenseDescriptor = Owner<cusparseDnMatDescr_t, cusparseDestroyDnMat>;
        using LegacyDescriptor = Owner<cusparseMatDescr_t, cusparseDestroyMatDescr>;
        using ProductDescriptor = Owner<cusparseSpGEMMDescr_t, cusparseSpGEMM_destroyDescr>;

        /** @brief 选择 CUDA 浮点类型，无参数、无额外约束。@tparam T 精度。@return CUDA 类型。
         * @code auto type = dataType<double>(); @endcode
         */
        template <Scalar T>
        constexpr cudaDataType dataType() {
            if constexpr (std::same_as<T, float>) {
                return CUDA_R_32F;
            }
            else {
                return CUDA_R_64F;
            }
        }
        /** @brief 下载设备缓冲区。@tparam T 元素类型。@param buffer 来源。@pre 同步已完成。
         * @return 主机数组。@code auto values = download(buffer); @endcode
         */
        template <BufferElement T>
        std::vector<T> download(const CudaBuffer<T>& buffer) {
            std::vector<T> values(buffer.size());
            buffer.copyToHost(values);
            return values;
        }
        /** @brief 初始化规范化 CSR 描述符。@tparam T 精度。@tparam I 索引类型。
         * @param descriptor 目标。@param matrix 设备矩阵。@pre 描述符为空，矩阵为 global。
         * @note 无返回值。@code createSparse(descriptor, matrix); @endcode
         */
        template <Scalar T, CsrIndex I>
        void createSparse(SparseDescriptor& descriptor, const CsrMatrix<T, I>& matrix) {
            constexpr auto indexType = sizeof(I) == 4 ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I;
            check(cusparseCreateCsr(descriptor.address(), matrix.rowCount(), matrix.columnCount(),
                                    matrix.nonZeroCount(), const_cast<I*>(matrix.rowOffsets().deviceData()),
                                    const_cast<I*>(matrix.columnIndices().deviceData()),
                                    const_cast<T*>(matrix.deviceValues()), indexType, indexType,
                                    CUSPARSE_INDEX_BASE_ZERO, dataType<T>()),
                  "cusparseCreateCsr");
        }
    } // namespace

    OperationScope::OperationScope(int device) {
        if (device >= 0) {
            check(cudaGetDevice(&previousDevice_), "cudaGetDevice");
            check(cudaSetDevice(device), "cudaSetDevice");
        }
    }
    OperationScope::~OperationScope() noexcept {
        if (previousDevice_ >= 0) {
            (void)cudaSetDevice(previousDevice_);
        }
    }

    template <Scalar T>
    void blasScale(int count, T alpha, T* values) {
        BlasHandle handle;
        check(cublasCreate(handle.address()), "cublasCreate");
        if constexpr (std::same_as<T, float>) {
            check(cublasSscal(handle.get(), count, &alpha, values, 1), "cublasSscal");
        }
        else {
            check(cublasDscal(handle.get(), count, &alpha, values, 1), "cublasDscal");
        }
        check(cudaStreamSynchronize(nullptr), "cuBLAS synchronize");
    }
    template <Scalar T>
    void blasAxpy(int count, T alpha, const T* x, T* y) {
        BlasHandle handle;
        check(cublasCreate(handle.address()), "cublasCreate");
        if constexpr (std::same_as<T, float>) {
            check(cublasSaxpy(handle.get(), count, &alpha, x, 1, y, 1), "cublasSaxpy");
        }
        else {
            check(cublasDaxpy(handle.get(), count, &alpha, x, 1, y, 1), "cublasDaxpy");
        }
        check(cudaStreamSynchronize(nullptr), "cuBLAS synchronize");
    }
    template <Scalar T>
    T blasDot(int count, const T* x, const T* y) {
        BlasHandle handle;
        check(cublasCreate(handle.address()), "cublasCreate");
        T result{};
        if constexpr (std::same_as<T, float>) {
            check(cublasSdot(handle.get(), count, x, 1, y, 1, &result), "cublasSdot");
        }
        else {
            check(cublasDdot(handle.get(), count, x, 1, y, 1, &result), "cublasDdot");
        }
        check(cudaStreamSynchronize(nullptr), "cuBLAS synchronize");
        return result;
    }
    template <Scalar T>
    T blasNorm(int count, const T* x, bool oneNorm) {
        BlasHandle handle;
        check(cublasCreate(handle.address()), "cublasCreate");
        T result{};
        if constexpr (std::same_as<T, float>) {
            check(oneNorm ? cublasSasum(handle.get(), count, x, 1, &result)
                          : cublasSnrm2(handle.get(), count, x, 1, &result),
                  "cuBLAS norm");
        }
        else {
            check(oneNorm ? cublasDasum(handle.get(), count, x, 1, &result)
                          : cublasDnrm2(handle.get(), count, x, 1, &result),
                  "cuBLAS norm");
        }
        check(cudaStreamSynchronize(nullptr), "cuBLAS synchronize");
        return result;
    }
    template <Scalar T>
    void blasTranspose(int rows, int columns, const T* input, T* output) {
        BlasHandle handle;
        check(cublasCreate(handle.address()), "cublasCreate");
        const T alpha{1}, beta{};
        if constexpr (std::same_as<T, float>) {
            check(cublasSgeam(handle.get(), CUBLAS_OP_T, CUBLAS_OP_T, rows, columns, &alpha, input, columns, &beta,
                              input, columns, output, rows),
                  "cublasSgeam");
        }
        else {
            check(cublasDgeam(handle.get(), CUBLAS_OP_T, CUBLAS_OP_T, rows, columns, &alpha, input, columns, &beta,
                              input, columns, output, rows),
                  "cublasDgeam");
        }
        check(cudaStreamSynchronize(nullptr), "cuBLAS synchronize");
    }
    template <Scalar T>
    void blasGemv(int rows, int columns, const T* matrix, const T* x, T* y) {
        BlasHandle handle;
        check(cublasCreate(handle.address()), "cublasCreate");
        const T alpha{1}, beta{};
        if constexpr (std::same_as<T, float>) {
            check(cublasSgemv(handle.get(), CUBLAS_OP_T, columns, rows, &alpha, matrix, columns, x, 1, &beta, y, 1),
                  "cublasSgemv");
        }
        else {
            check(cublasDgemv(handle.get(), CUBLAS_OP_T, columns, rows, &alpha, matrix, columns, x, 1, &beta, y, 1),
                  "cublasDgemv");
        }
        check(cudaStreamSynchronize(nullptr), "cuBLAS synchronize");
    }
    template <Scalar T>
    void blasGemm(int rows, int columns, int inner, const T* a, const T* b, T* c) {
        BlasHandle handle;
        check(cublasCreate(handle.address()), "cublasCreate");
        check(cublasSetMathMode(handle.get(), CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
        const T alpha{1}, beta{};
        if constexpr (std::same_as<T, float>) {
            check(cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, columns, rows, inner, &alpha, b, columns, a,
                              inner, &beta, c, columns),
                  "cublasSgemm");
        }
        else {
            check(cublasDgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, columns, rows, inner, &alpha, b, columns, a,
                              inner, &beta, c, columns),
                  "cublasDgemm");
        }
        check(cudaStreamSynchronize(nullptr), "cuBLAS synchronize");
    }

    template <Scalar T, CsrIndex I>
    void sparseMv(const CsrMatrix<T, I>& matrix, const T* x, T* y) {
        SparseHandle handle;
        check(cusparseCreate(handle.address()), "cusparseCreate");
        SparseDescriptor a;
        VectorDescriptor input, output;
        createSparse(a, matrix);
        check(cusparseCreateDnVec(input.address(), matrix.columnCount(), const_cast<T*>(x), dataType<T>()),
              "cusparseCreateDnVec");
        check(cusparseCreateDnVec(output.address(), matrix.rowCount(), y, dataType<T>()), "cusparseCreateDnVec");
        const T alpha{1}, beta{};
        std::size_t bytes{};
        check(cusparseSpMV_bufferSize(handle.get(), CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, a.get(), input.get(),
                                      &beta, output.get(), dataType<T>(), CUSPARSE_SPMV_ALG_DEFAULT, &bytes),
              "cusparseSpMV_bufferSize");
        CudaBuffer<std::byte> workspace(bytes);
        check(cusparseSpMV(handle.get(), CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, a.get(), input.get(), &beta,
                           output.get(), dataType<T>(), CUSPARSE_SPMV_ALG_DEFAULT, workspace.data()),
              "cusparseSpMV");
        check(cudaStreamSynchronize(nullptr), "cuSPARSE synchronize");
    }
    template <Scalar T, CsrIndex I>
    void sparseMm(const CsrMatrix<T, I>& matrix, std::size_t columns, const T* b, T* c) {
        SparseHandle handle;
        check(cusparseCreate(handle.address()), "cusparseCreate");
        SparseDescriptor a;
        DenseDescriptor input, output;
        createSparse(a, matrix);
        check(cusparseCreateDnMat(input.address(), matrix.columnCount(), columns, columns, const_cast<T*>(b),
                                  dataType<T>(), CUSPARSE_ORDER_ROW),
              "cusparseCreateDnMat");
        check(cusparseCreateDnMat(output.address(), matrix.rowCount(), columns, columns, c, dataType<T>(),
                                  CUSPARSE_ORDER_ROW),
              "cusparseCreateDnMat");
        const T alpha{1}, beta{};
        std::size_t bytes{};
        check(cusparseSpMM_bufferSize(handle.get(), CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha, a.get(), input.get(), &beta, output.get(), dataType<T>(),
                                      CUSPARSE_SPMM_ALG_DEFAULT, &bytes),
              "cusparseSpMM_bufferSize");
        CudaBuffer<std::byte> workspace(bytes);
        check(cusparseSpMM(handle.get(), CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                           a.get(), input.get(), &beta, output.get(), dataType<T>(), CUSPARSE_SPMM_ALG_DEFAULT,
                           workspace.data()),
              "cusparseSpMM");
        check(cudaStreamSynchronize(nullptr), "cuSPARSE synchronize");
    }

    template <Scalar T>
    SparseHostData<T> sparseAdd(const CsrMatrix<T, int>& a, const CsrMatrix<T, int>& b, T beta) {
        if (a.nonZeroCount() > static_cast<std::size_t>(std::numeric_limits<int>::max()) - b.nonZeroCount()) {
            throw std::length_error("Sparse addition may exceed 32-bit structural range");
        }
        SparseHandle handle;
        check(cusparseCreate(handle.address()), "cusparseCreate");
        LegacyDescriptor descriptor;
        check(cusparseCreateMatDescr(descriptor.address()), "cusparseCreateMatDescr");
        const int rows = static_cast<int>(a.rowCount()), columns = static_cast<int>(a.columnCount());
        const int countA = static_cast<int>(a.nonZeroCount()), countB = static_cast<int>(b.nonZeroCount());
        const T alpha{1};
        CudaBuffer<int> offsets(a.rowCount() + 1);
        std::size_t bytes{};
        if constexpr (std::same_as<T, float>) {
            check(cusparseScsrgeam2_bufferSizeExt(
                      handle.get(), rows, columns, &alpha, descriptor.get(), countA, a.deviceValues(),
                      a.rowOffsets().deviceData(), a.columnIndices().deviceData(), &beta, descriptor.get(), countB,
                      b.deviceValues(), b.rowOffsets().deviceData(), b.columnIndices().deviceData(), descriptor.get(),
                      nullptr, offsets.data(), nullptr, &bytes),
                  "cusparseScsrgeam2_bufferSizeExt");
        }
        else {
            check(cusparseDcsrgeam2_bufferSizeExt(
                      handle.get(), rows, columns, &alpha, descriptor.get(), countA, a.deviceValues(),
                      a.rowOffsets().deviceData(), a.columnIndices().deviceData(), &beta, descriptor.get(), countB,
                      b.deviceValues(), b.rowOffsets().deviceData(), b.columnIndices().deviceData(), descriptor.get(),
                      nullptr, offsets.data(), nullptr, &bytes),
                  "cusparseDcsrgeam2_bufferSizeExt");
        }
        CudaBuffer<std::byte> workspace(bytes);
        int count{};
        check(cusparseXcsrgeam2Nnz(handle.get(), rows, columns, descriptor.get(), countA, a.rowOffsets().deviceData(),
                                   a.columnIndices().deviceData(), descriptor.get(), countB,
                                   b.rowOffsets().deviceData(), b.columnIndices().deviceData(), descriptor.get(),
                                   offsets.data(), &count, workspace.data()),
              "cusparseXcsrgeam2Nnz");
        CudaBuffer<int> indices(static_cast<std::size_t>(count));
        CudaBuffer<T> values(static_cast<std::size_t>(count));
        if (count > 0) {
            if constexpr (std::same_as<T, float>) {
                check(cusparseScsrgeam2(handle.get(), rows, columns, &alpha, descriptor.get(), countA, a.deviceValues(),
                                        a.rowOffsets().deviceData(), a.columnIndices().deviceData(), &beta,
                                        descriptor.get(), countB, b.deviceValues(), b.rowOffsets().deviceData(),
                                        b.columnIndices().deviceData(), descriptor.get(), values.data(), offsets.data(),
                                        indices.data(), workspace.data()),
                      "cusparseScsrgeam2");
            }
            else {
                check(cusparseDcsrgeam2(handle.get(), rows, columns, &alpha, descriptor.get(), countA, a.deviceValues(),
                                        a.rowOffsets().deviceData(), a.columnIndices().deviceData(), &beta,
                                        descriptor.get(), countB, b.deviceValues(), b.rowOffsets().deviceData(),
                                        b.columnIndices().deviceData(), descriptor.get(), values.data(), offsets.data(),
                                        indices.data(), workspace.data()),
                      "cusparseDcsrgeam2");
            }
        }
        check(cudaStreamSynchronize(nullptr), "cuSPARSE synchronize");
        return {download(offsets), download(indices), download(values)};
    }

    template <Scalar T>
    SparseHostData<T> sparseTranspose(const CsrMatrix<T, int>& matrix) {
        SparseHandle handle;
        check(cusparseCreate(handle.address()), "cusparseCreate");
        CudaBuffer<int> offsets(matrix.columnCount() + 1), indices(matrix.nonZeroCount());
        CudaBuffer<T> values(matrix.nonZeroCount());
        const auto rows = static_cast<int>(matrix.rowCount()), columns = static_cast<int>(matrix.columnCount());
        const auto count = static_cast<int>(matrix.nonZeroCount());
        std::size_t bytes{};
        check(cusparseCsr2cscEx2_bufferSize(
                  handle.get(), rows, columns, count, matrix.deviceValues(), matrix.rowOffsets().deviceData(),
                  matrix.columnIndices().deviceData(), values.data(), offsets.data(), indices.data(), dataType<T>(),
                  CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &bytes),
              "cusparseCsr2cscEx2_bufferSize");
        CudaBuffer<std::byte> workspace(bytes);
        check(cusparseCsr2cscEx2(handle.get(), rows, columns, count, matrix.deviceValues(),
                                 matrix.rowOffsets().deviceData(), matrix.columnIndices().deviceData(), values.data(),
                                 offsets.data(), indices.data(), dataType<T>(), CUSPARSE_ACTION_NUMERIC,
                                 CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, workspace.data()),
              "cusparseCsr2cscEx2");
        check(cudaStreamSynchronize(nullptr), "cuSPARSE synchronize");
        return {download(offsets), download(indices), download(values)};
    }

    template <Scalar T>
    SparseHostData<T> sparseGemm(const CsrMatrix<T, int>& a, const CsrMatrix<T, int>& b) {
        SparseHandle handle;
        check(cusparseCreate(handle.address()), "cusparseCreate");
        SparseDescriptor left, right, output;
        createSparse(left, a);
        createSparse(right, b);
        check(cusparseCreateCsr(output.address(), a.rowCount(), b.columnCount(), 0, nullptr, nullptr, nullptr,
                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, dataType<T>()),
              "cusparseCreateCsr");
        ProductDescriptor product;
        check(cusparseSpGEMM_createDescr(product.address()), "cusparseSpGEMM_createDescr");
        const T alpha{1}, beta{};
        constexpr auto op = CUSPARSE_OPERATION_NON_TRANSPOSE;
        constexpr auto algorithm = CUSPARSE_SPGEMM_DEFAULT;
        const auto estimationContext = std::string("cusparseSpGEMM_workEstimation [m=") + std::to_string(a.rowCount()) +
            ", k=" + std::to_string(a.columnCount()) + ", n=" + std::to_string(b.columnCount()) +
            ", nnzA=" + std::to_string(a.nonZeroCount()) + ", nnzB=" + std::to_string(b.nonZeroCount()) + "]";
        std::size_t bytes1{}, bytes2{};
        check(cusparseSpGEMM_workEstimation(handle.get(), op, op, &alpha, left.get(), right.get(), &beta, output.get(),
                                            dataType<T>(), algorithm, product.get(), &bytes1, nullptr),
              estimationContext.c_str());
        CudaBuffer<std::byte> workspace1(bytes1);
        check(cusparseSpGEMM_workEstimation(handle.get(), op, op, &alpha, left.get(), right.get(), &beta, output.get(),
                                            dataType<T>(), algorithm, product.get(), &bytes1, workspace1.data()),
              estimationContext.c_str());
        check(cusparseSpGEMM_compute(handle.get(), op, op, &alpha, left.get(), right.get(), &beta, output.get(),
                                     dataType<T>(), algorithm, product.get(), &bytes2, nullptr),
              "cusparseSpGEMM_compute");
        CudaBuffer<std::byte> workspace2(bytes2);
        check(cusparseSpGEMM_compute(handle.get(), op, op, &alpha, left.get(), right.get(), &beta, output.get(),
                                     dataType<T>(), algorithm, product.get(), &bytes2, workspace2.data()),
              "cusparseSpGEMM_compute");
        std::int64_t rows{}, columns{}, count{};
        check(cusparseSpMatGetSize(output.get(), &rows, &columns, &count), "cusparseSpMatGetSize");
        if (!std::in_range<int>(count)) {
            throw std::length_error("Sparse product exceeds 32-bit structural index range");
        }
        CudaBuffer<int> offsets(a.rowCount() + 1), indices(static_cast<std::size_t>(count));
        CudaBuffer<T> values(static_cast<std::size_t>(count));
        check(cusparseCsrSetPointers(output.get(), offsets.data(), indices.data(), values.data()),
              "cusparseCsrSetPointers");
        check(cusparseSpGEMM_copy(handle.get(), op, op, &alpha, left.get(), right.get(), &beta, output.get(),
                                  dataType<T>(), algorithm, product.get()),
              "cusparseSpGEMM_copy");
        check(cudaStreamSynchronize(nullptr), "cuSPARSE synchronize");
        return {download(offsets), download(indices), download(values)};
    }

// 后端仅为受支持的精度及索引组合生成链接符号。
#define THERMOFEM_INSTANTIATE(T)                                                                                       \
    template void blasScale<T>(int, T, T*);                                                                            \
    template void blasAxpy<T>(int, T, const T*, T*);                                                                   \
    template T blasDot<T>(int, const T*, const T*);                                                                    \
    template T blasNorm<T>(int, const T*, bool);                                                                       \
    template void blasTranspose<T>(int, int, const T*, T*);                                                            \
    template void blasGemv<T>(int, int, const T*, const T*, T*);                                                       \
    template void blasGemm<T>(int, int, int, const T*, const T*, T*);                                                  \
    template void sparseMv<T, std::int32_t>(const CsrMatrix<T, std::int32_t>&, const T*, T*);                          \
    template void sparseMv<T, std::int64_t>(const CsrMatrix<T, std::int64_t>&, const T*, T*);                          \
    template void sparseMm<T, std::int32_t>(const CsrMatrix<T, std::int32_t>&, std::size_t, const T*, T*);             \
    template void sparseMm<T, std::int64_t>(const CsrMatrix<T, std::int64_t>&, std::size_t, const T*, T*);             \
    template SparseHostData<T> sparseAdd<T>(const CsrMatrix<T, int>&, const CsrMatrix<T, int>&, T);                    \
    template SparseHostData<T> sparseTranspose<T>(const CsrMatrix<T, int>&);                                           \
    template SparseHostData<T> sparseGemm<T>(const CsrMatrix<T, int>&, const CsrMatrix<T, int>&);
    THERMOFEM_INSTANTIATE(float)
    THERMOFEM_INSTANTIATE(double)
#undef THERMOFEM_INSTANTIATE
} // namespace thermofem::linalg::detail

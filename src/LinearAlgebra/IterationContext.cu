/** @file IterationContext.cu
 * @brief 实现缓存句柄、SpMV 描述符及 Jacobi CUDA 更新内核。
 */
#include <LinearAlgebra/IterationContext.cuh>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cmath>

namespace thermofem::linalg {
/** @brief 提供执行上下文的内部错误检查与 CUDA 内核。 */
namespace {
/** @brief 检查库状态。@tparam S 状态枚举。@param status 返回状态。
 * @pre 零表示成功。@return 无返回值。@code check(cublasCreate(&handle)); @endcode
 */
template<class S> requires std::is_enum_v<S>
void check(S status) {
    if (static_cast<int>(status) != 0) { throw std::runtime_error("CUDA iteration library failure: " + std::to_string(static_cast<int>(status))); }
}
/** @brief 逐元素更新 Jacobi 近似解。@tparam T 精度。@param n 长度。@param w 权重。
 * @param d 逆对角线。@param b 右端项。@param a 矩阵向量积。@param x 近似解。
 * @pre 设备地址有效且等长。@return 无返回值。@code update<<<1,256>>>(n,w,d,b,a,x); @endcode
 */
template<Scalar T> __global__ void update(int n, T w, const T* d, const T* b, const T* a, T* x) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) { x[i] += w * d[i] * (b[i] - a[i]); }
}
}

/** @brief 独占库资源及矩阵快照，释放时恢复调用方设备。
 * @code IterationContext<double>::Impl state(matrix); @endcode
 */
template<Scalar T, CsrIndex I>
struct IterationContext<T, I>::Impl {
    CsrMatrix<T, I> matrix_; ///< 规范化 global 矩阵。
    Vector<T> input_, output_; ///< 描述符初始占位存储。
    CudaBuffer<std::byte> workspace_; ///< SpMV 工作区。
    cublasHandle_t blas_{}; ///< 持久化 BLAS 句柄。
    cusparseHandle_t sparse_{}; ///< 持久化稀疏句柄。
    cusparseSpMatDescr_t a_{}; ///< 矩阵描述符。
    cusparseDnVecDescr_t x_{}, y_{}; ///< 可重绑定的向量描述符。
    int device_{}; ///< 资源所属设备。
    /** @brief 建立所有资源。@param matrix 输入。
     * @pre 当前设备正确。@return 无返回值。@code Impl state(matrix); @endcode
     */
    explicit Impl(const CsrMatrix<T, I>& matrix) : matrix_(detail::canonical<I>(matrix)),
        input_(matrix.columnCount()), output_(matrix.rowCount()) {
        check(cudaGetDevice(&device_));
        for (auto value : detail::hostCopy(matrix_.values())) {
            if (!std::isfinite(value)) { throw std::invalid_argument("Matrix contains nonfinite entries"); }
        }
        try {
            check(cublasCreate(&blas_));
            check(cublasSetMathMode(blas_, CUBLAS_PEDANTIC_MATH));
            check(cusparseCreate(&sparse_));
            if (matrix_.rowCount() && matrix_.columnCount() && matrix_.nonZeroCount()) {
                const auto index = sizeof(I) == 4 ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I;
                const auto type = std::same_as<T, float> ? CUDA_R_32F : CUDA_R_64F;
                check(cusparseCreateCsr(&a_, matrix_.rowCount(), matrix_.columnCount(), matrix_.nonZeroCount(),
                    const_cast<I*>(matrix_.rowOffsets().deviceData()), const_cast<I*>(matrix_.columnIndices().deviceData()),
                    const_cast<T*>(matrix_.values().deviceData()), index, index, CUSPARSE_INDEX_BASE_ZERO, type));
                check(cusparseCreateDnVec(&x_, input_.size(), input_.deviceData(), type));
                check(cusparseCreateDnVec(&y_, output_.size(), output_.deviceData(), type));
                const T one = 1, zero = 0;
                std::size_t size{};
                check(cusparseSpMV_bufferSize(sparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, a_, x_, &zero, y_,
                    type, CUSPARSE_SPMV_ALG_DEFAULT, &size));
                workspace_ = CudaBuffer<std::byte>(size);
            }
        } catch (...) { release(); throw; }
    }
    /** @brief 释放句柄，无参数、无额外约束。@return 无返回值。
     * @code state.release(); @endcode
     */
    void release() noexcept {
        if (y_) { cusparseDestroyDnVec(y_); }
        if (x_) { cusparseDestroyDnVec(x_); }
        if (a_) { cusparseDestroySpMat(a_); }
        if (sparse_) { cusparseDestroy(sparse_); }
        if (blas_) { cublasDestroy(blas_); }
    }
    /** @brief 在所属设备释放资源，无参数、无额外约束。@return 无返回值。
     * @code { Impl state(matrix); } @endcode
     */
    ~Impl() noexcept {
        int previous{}; cudaGetDevice(&previous); cudaSetDevice(device_);
        release(); cudaSetDevice(previous);
    }
    /** @brief 检查工作向量。@param v 向量。@pre 无额外约束。
     * @return 无返回值。@code state.validate(v); @endcode
     */
    void validate(const Vector<T>& v) const {
        detail::librarySize(v.size());
        if (v.memoryType() != MemoryType::Global || (v.size() && v.deviceId() != device_)) {
            throw std::invalid_argument("Iteration workspace requires same-device global vectors");
        }
    }
};

template<Scalar T, CsrIndex I>
IterationContext<T, I>::IterationContext(const CsrMatrix<T, I>& matrix) {
    detail::librarySize(matrix.rowCount()); detail::librarySize(matrix.columnCount());
    detail::OperationScope scope(matrix.values().deviceId());
    impl_ = std::make_unique<Impl>(matrix);
}
template<Scalar T, CsrIndex I> IterationContext<T, I>::~IterationContext() noexcept = default;
template<Scalar T, CsrIndex I>
const CsrMatrix<T, I>& IterationContext<T, I>::matrix() const noexcept { return impl_->matrix_; }
template<Scalar T, CsrIndex I>
void IterationContext<T, I>::multiply(const Vector<T>& x, Vector<T>& y) {
    impl_->validate(x); impl_->validate(y);
    if (x.size() != impl_->matrix_.columnCount() || y.size() != impl_->matrix_.rowCount()
        || (x.size() && x.data() == y.data())) { throw std::invalid_argument("SpMV shape or alias mismatch"); }
    detail::OperationScope scope(impl_->device_);
    if (!impl_->a_) { detail::zero(y.deviceData(), y.size()); return; }
    check(cusparseDnVecSetValues(impl_->x_, const_cast<T*>(x.deviceData())));
    check(cusparseDnVecSetValues(impl_->y_, y.deviceData()));
    const T one = 1, zero = 0;
    check(cusparseSpMV(impl_->sparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, impl_->a_, impl_->x_,
        &zero, impl_->y_, std::same_as<T, float> ? CUDA_R_32F : CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, impl_->workspace_.deviceData()));
    check(cudaStreamSynchronize(nullptr));
}
template<Scalar T, CsrIndex I>
void IterationContext<T, I>::axpy(T alpha, const Vector<T>& x, Vector<T>& y) {
    impl_->validate(x); impl_->validate(y);
    if (x.size() != y.size()) { throw std::invalid_argument("AXPY shape mismatch"); }
    if (x.empty()) { return; }
    detail::OperationScope scope(impl_->device_);
    if constexpr (std::same_as<T, float>) { check(cublasSaxpy(impl_->blas_, detail::librarySize(x.size()), &alpha, x.deviceData(), 1, y.deviceData(), 1)); }
    else { check(cublasDaxpy(impl_->blas_, detail::librarySize(x.size()), &alpha, x.deviceData(), 1, y.deviceData(), 1)); }
    check(cudaStreamSynchronize(nullptr));
}
template<Scalar T, CsrIndex I>
void IterationContext<T, I>::scale(T alpha, Vector<T>& x) {
    impl_->validate(x); if (x.empty()) { return; }
    detail::OperationScope scope(impl_->device_);
    if constexpr (std::same_as<T, float>) { check(cublasSscal(impl_->blas_, detail::librarySize(x.size()), &alpha, x.deviceData(), 1)); }
    else { check(cublasDscal(impl_->blas_, detail::librarySize(x.size()), &alpha, x.deviceData(), 1)); }
    check(cudaStreamSynchronize(nullptr));
}
template<Scalar T, CsrIndex I>
T IterationContext<T, I>::dot(const Vector<T>& x, const Vector<T>& y) {
    impl_->validate(x); impl_->validate(y);
    if (x.size() != y.size()) { throw std::invalid_argument("Dot shape mismatch"); }
    T result{}; if (x.empty()) { return result; }
    detail::OperationScope scope(impl_->device_);
    if constexpr (std::same_as<T, float>) { check(cublasSdot(impl_->blas_, detail::librarySize(x.size()), x.deviceData(), 1, y.deviceData(), 1, &result)); }
    else { check(cublasDdot(impl_->blas_, detail::librarySize(x.size()), x.deviceData(), 1, y.deviceData(), 1, &result)); }
    return result;
}
template<Scalar T, CsrIndex I>
T IterationContext<T, I>::norm(const Vector<T>& x) {
    impl_->validate(x); T result{}; if (x.empty()) { return result; }
    detail::OperationScope scope(impl_->device_);
    if constexpr (std::same_as<T, float>) { check(cublasSnrm2(impl_->blas_, detail::librarySize(x.size()), x.deviceData(), 1, &result)); }
    else { check(cublasDnrm2(impl_->blas_, detail::librarySize(x.size()), x.deviceData(), 1, &result)); }
    return result;
}
template<Scalar T, CsrIndex I>
void IterationContext<T, I>::jacobi(T weight, const Vector<T>& inverse, const Vector<T>& rhs,
                                  const Vector<T>& product, Vector<T>& x) {
    impl_->validate(inverse); impl_->validate(rhs); impl_->validate(product); impl_->validate(x);
    if (inverse.size() != x.size() || rhs.size() != x.size() || product.size() != x.size()) {
        throw std::invalid_argument("Jacobi shape mismatch");
    }
    if (x.empty()) { return; }
    detail::OperationScope scope(impl_->device_);
    const int n = detail::librarySize(x.size());
    update<<<(static_cast<unsigned>(n) + 255) / 256, 256>>>(n, weight, inverse.deviceData(), rhs.deviceData(), product.deviceData(), x.deviceData());
    check(cudaGetLastError()); check(cudaStreamSynchronize(nullptr));
}
template class IterationContext<float, std::int32_t>;
template class IterationContext<double, std::int32_t>;
template class IterationContext<float, std::int64_t>;
template class IterationContext<double, std::int64_t>;
} // namespace thermofem::linalg

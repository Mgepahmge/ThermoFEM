/** @file IterationContext.cuh
 * @brief 定义迭代计算使用的持久化 CUDA 库句柄、稀疏描述符和工作区。
 */
#pragma once
#include <LinearAlgebra/Operations.cuh>
#include <memory>

namespace thermofem::linalg {
/** @brief 持有规范化矩阵快照，复用 cuBLAS/cuSPARSE 资源；不可复制。
 * @tparam T 浮点类型。@tparam I CSR 索引。
 * @details 输入工作向量必须为相同设备上的 global 向量。方法为同步操作。
 * @code IterationContext<double> context(matrix); context.multiply(x, y); @endcode
 */
template<Scalar T, CsrIndex I = int>
class IterationContext {
public:
    /** @brief 建立矩阵快照及库资源。@param matrix 有限值 CSR。
     * @pre CUDA 设备可用，维度不超过 INT_MAX。@return 无返回值。
     * @code IterationContext<double> context(matrix); @endcode
     */
    explicit IterationContext(const CsrMatrix<T, I>& matrix);
    /** @brief 释放所属设备上的资源，无参数、无额外约束。@return 无返回值。
     * @code { IterationContext<double> context(matrix); } @endcode
     */
    ~IterationContext() noexcept;
    IterationContext(const IterationContext&) = delete; ///< 禁止复制库资源。
    IterationContext& operator=(const IterationContext&) = delete; ///< 禁止复制库资源。
    /** @brief 获取矩阵快照，无参数、无额外约束。@return 只读矩阵。
     * @code auto n = context.matrix().rowCount(); @endcode
     */
    const CsrMatrix<T, I>& matrix() const noexcept;
    /** @brief 计算 y=A*x。@param x 输入。@param y 输出。
     * @pre 维度匹配，不别名，global 存储且同设备。@return 无返回值。
     * @code context.multiply(x, y); @endcode
     */
    void multiply(const Vector<T>& x, Vector<T>& y);
    /** @brief 执行 y+=alpha*x。@param alpha 系数。@param x 输入。@param y 输出。
     * @pre 等长 global 向量，同设备。@return 无返回值。@code context.axpy(1, x, y); @endcode
     */
    void axpy(T alpha, const Vector<T>& x, Vector<T>& y);
    /** @brief 执行 x*=alpha。@param alpha 系数。@param x 输入输出。
     * @pre global 向量，同设备。@return 无返回值。@code context.scale(2, x); @endcode
     */
    void scale(T alpha, Vector<T>& x);
    /** @brief 计算点积。@param x 输入。@param y 输入。
     * @pre 等长 global 向量，同设备。@return 点积。@code auto d = context.dot(x, y); @endcode
     */
    T dot(const Vector<T>& x, const Vector<T>& y);
    /** @brief 计算二范数。@param x 输入。@pre global 向量，同设备。
     * @return 二范数。@code auto n = context.norm(x); @endcode
     */
    T norm(const Vector<T>& x);
    /** @brief 执行 Jacobi 更新 x+=weight*inverse*(rhs-product)。
     * @param weight 松弛系数。@param inverse 逆对角线。@param rhs 右端项。
     * @param product 已计算的 A*x。@param x 输入输出。
     * @pre 等长 global 向量且同设备。@return 无返回值。
     * @code context.jacobi(0.7, inverse, rhs, product, x); @endcode
     */
    void jacobi(T weight, const Vector<T>& inverse, const Vector<T>& rhs,
                const Vector<T>& product, Vector<T>& x);
private:
    struct Impl; ///< CUDA 资源实现。
    std::unique_ptr<Impl> impl_; ///< 独占矩阵及执行资源。
};
} // namespace thermofem::linalg

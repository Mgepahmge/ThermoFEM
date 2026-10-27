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
     * @par 资源与复用
     * 构造时规范化并深拷贝矩阵，分配 cuBLAS/cuSPARSE 句柄、SpMV 描述符、占位向量和工作区。
     * 后续 multiply 仅重新绑定输入输出地址，不重新分配描述符或工作区。
     * 本类通过组合拥有这些资源，不继承矩阵；调用者可在构造完成后销毁原矩阵。
     * @par 契约与同步
     * 这是面向迭代缓存的低层接口，主动要求 global 向量；高层组件负责其他策略的暂存。
     * 默认流执行并同步返回，自动选择所属设备并恢复调用方设备。描述符绑定可变，
     * 因而同一实例不可并发调用。multiply/residual 输出不允许与输入别名。
     * @par 适用范围
     * 支持矩形 SpMV，向量长度受 cuBLAS 的 int 接口限制。零非零数矩阵走清零路径。
     * 不包含收敛判断、平滑次数或多层递归，所有算法控制属于调用组件。
     * @code IterationContext<double> context(matrix); context.multiply(x, y); @endcode
     */
    template <Scalar T, CsrIndex I = int>
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
        /** @brief 计算真实残差 out=rhs-A*x。@param rhs 右端项。@param x 当前解。@param out 输出。
         * @pre 维度匹配，同设备 global 向量，输出不与任一输入别名。@return 无返回值。
         * @code context.residual(rhs, x, out); @endcode
         */
        void residual(const Vector<T>& rhs, const Vector<T>& x, Vector<T>& out);
        /** @brief 将工作向量清零。@param x 输入输出。@pre 同设备 global 向量。
         * @return 无返回值。@code context.clear(x); @endcode
         */
        void clear(Vector<T>& x);
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
        void jacobi(T weight, const Vector<T>& inverse, const Vector<T>& rhs, const Vector<T>& product, Vector<T>& x);

    private:
        struct Impl; ///< CUDA 资源实现。
        std::unique_ptr<Impl> impl_; ///< 独占矩阵及执行资源。
    };
} // namespace thermofem::linalg

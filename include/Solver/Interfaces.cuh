/** @file Interfaces.cuh
 * @brief 定义平滑器、预条件子、求解器三个独立运行时接口及求解报告。
 */
#pragma once
#include <LinearAlgebra/IterationContext.cuh>

/** @brief 提供解耦的平滑、预条件与迭代求解组件。 */
namespace thermofem {
/** @brief 标记求解终止原因。 */
enum class SolveStatus { Converged, MaxIterations, Breakdown, NonFinite };
/** @brief 保存真实残差和终止原因。
 * @code auto result = solver.apply(rhs, x); @endcode
 */
template<linalg::Scalar T> struct SolveResult {
    SolveStatus status_{SolveStatus::MaxIterations}; ///< 终止状态。
    int iterations_{}; ///< 已执行迭代次数。
    T residualNorm_{}; ///< 最终真实残差二范数。
    T relativeResidual_{}; ///< 相对右端项的残差；零右端项时等于绝对残差。
};
/** @brief 迭代平滑器接口，apply 从传入初值更新近似解。
 * @code std::unique_ptr<Smoother<double>> smoother = std::make_unique<Jacobi<double>>(); @endcode
 */
template<linalg::Scalar T, CsrIndex I = int> class Smoother {
public:
    /** @brief 多态释放资源，无参数、无额外约束。@return 无返回值。
     * @code smoother.reset(); @endcode
     */
    virtual ~Smoother() = default;
    /** @brief 保存矩阵并建立缓存。@param matrix 系数矩阵。
     * @pre 非空有限值方阵。@return 无返回值。@code smoother.setup(matrix); @endcode
     */
    virtual void setup(const CsrMatrix<T, I>& matrix) = 0;
    /** @brief 对初值执行平滑。@param rhs 右端项。@param x 输入输出近似解。
     * @pre setup 成功，尺寸设备匹配。@return 无返回值。@code smoother.apply(rhs, x); @endcode
     */
    virtual void apply(const Vector<T>& rhs, Vector<T>& x) = 0;
};
/** @brief 预条件接口，apply 覆盖输出，不依赖其原值。
 * @code std::unique_ptr<Preconditioner<double>> p = std::make_unique<Amg<double>>(); @endcode
 */
template<linalg::Scalar T, CsrIndex I = int> class Preconditioner {
public:
    /** @brief 多态释放资源，无参数、无额外约束。@return 无返回值。
     * @code p.reset(); @endcode
     */
    virtual ~Preconditioner() = default;
    /** @brief 建立预条件状态。@param matrix 系数矩阵。
     * @pre 非空有限值方阵。@return 无返回值。@code p.setup(matrix); @endcode
     */
    virtual void setup(const CsrMatrix<T, I>& matrix) = 0;
    /** @brief 近似求解并覆盖输出。@param rhs 输入。@param x 输出。
     * @pre setup 成功，尺寸设备匹配。@return 无返回值。@code p.apply(rhs, x); @endcode
     */
    virtual void apply(const Vector<T>& rhs, Vector<T>& x) = 0;
};
/** @brief 迭代求解接口，使用传入初值并返回明确的终止报告。
 * @code std::unique_ptr<Solver<double>> solver = std::make_unique<BiCgStab<double>>(); @endcode
 */
template<linalg::Scalar T, CsrIndex I = int> class Solver {
public:
    /** @brief 多态释放资源，无参数、无额外约束。@return 无返回值。
     * @code solver.reset(); @endcode
     */
    virtual ~Solver() = default;
    /** @brief 建立矩阵及预条件状态。@param matrix 系数矩阵。
     * @pre 非空有限值方阵。@return 无返回值。@code solver.setup(matrix); @endcode
     */
    virtual void setup(const CsrMatrix<T, I>& matrix) = 0;
    /** @brief 从初值求解 A*x=rhs。@param rhs 右端项。@param x 输入输出解。
     * @pre setup 成功，尺寸设备匹配。@return 终止报告。@code auto report = solver.apply(rhs, x); @endcode
     */
    virtual SolveResult<T> apply(const Vector<T>& rhs, Vector<T>& x) = 0;
};
/** @brief 提供求解组件共享的矩阵和向量前置条件检查。 */
namespace iteration {
/** @brief 检查非空方阵。@tparam T 精度。@tparam I 索引。@param matrix 输入。
 * @pre 无额外约束。@return 无返回值。@code validateMatrix(matrix); @endcode
 */
template<linalg::Scalar T, CsrIndex I> void validateMatrix(const CsrMatrix<T, I>& matrix) {
    if (!matrix.rowCount() || matrix.rowCount() != matrix.columnCount()) {
        throw std::invalid_argument("Solver requires a nonempty square matrix");
    }
}
/** @brief 检查向量尺寸与设备。@tparam T 精度。@param n 维度。@param device 所属设备。
 * @param rhs 右端项。@param x 解。@pre 无额外约束。@return 无返回值。
 * @code validateVectors(n, device, rhs, x); @endcode
 */
template<linalg::Scalar T> void validateVectors(std::size_t n, int device, const Vector<T>& rhs, const Vector<T>& x) {
    if (rhs.size() != n || x.size() != n) { throw std::invalid_argument("Solver vector shape mismatch"); }
    linalg::detail::commonDevice({device, rhs.deviceId(), x.deviceId()});
}
}
} // namespace thermofem

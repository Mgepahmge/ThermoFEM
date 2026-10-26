/** @file Backend.cuh
 * @brief 声明 cuBLAS、cuSPARSE 后端和运算设备守卫，供上层模板复用。
 */
#pragma once
#include <DataStructure/CsrMatrix.cuh>
#include <vector>

namespace thermofem::linalg {
/// 线性代数后端支持的实数精度。
template<class T>
concept Scalar = std::same_as<T, float> || std::same_as<T, double>;

/** @brief 提供线性代数内部设备管理、库调用与结果传输设施。 */
namespace detail {
/** @brief 临时选择运算设备，并在退出时恢复调用方设备；禁止复制。
 * @code OperationScope scope(vector.deviceId()); @endcode
 */
class OperationScope {
public:
    /** @brief 激活运算设备。@param device 设备编号，-1 表示不切换。
     * @pre 编号有效。@return 无返回值。@code OperationScope scope(0); @endcode
     */
    explicit OperationScope(int device);
    /** @brief 尽力恢复原设备，无参数、无额外约束。@return 无返回值。
     * @code { OperationScope scope(0); } @endcode
     */
    ~OperationScope() noexcept;
    OperationScope(const OperationScope&) = delete; ///< 禁止复制设备状态。
    OperationScope& operator=(const OperationScope&) = delete; ///< 禁止覆盖设备状态。
private:
    int previousDevice_{-1}; ///< 调用方设备，未切换时为 -1。
};

/** @brief 保存 cuSPARSE 生成的主机端 CSR 结果，结构索引使用 int。
 * @tparam T 浮点精度。
 * @code SparseHostData<double> result; @endcode
 */
template<Scalar T>
struct SparseHostData {
    std::vector<int> offsets_; ///< 按行排列的偏移。
    std::vector<int> columns_; ///< 非零元素列索引。
    std::vector<T> values_; ///< 非零元素数值。
};

/** @brief 缩放设备数据。@param count 元素数。@param alpha 系数。@param values 设备数据。
 * @pre count 非负，地址有效，当前设备正确。@return 无返回值。
 * @code blasScale(3, 2.0, pointer); @endcode
 */
template<Scalar T> void blasScale(int count, T alpha, T* values);
/** @brief 累加 alpha*x 到 y。@param count 元素数。@param alpha 系数。@param x 输入。@param y 输出。
 * @pre 地址有效且所属当前设备。@return 无返回值。@code blasAxpy(3, 2.0, x, y); @endcode
 */
template<Scalar T> void blasAxpy(int count, T alpha, const T* x, T* y);
/** @brief 计算点积。@param count 元素数。@param x 输入。@param y 输入。
 * @pre 地址有效且所属当前设备。@return 点积。@code auto value = blasDot(3, x, y); @endcode
 */
template<Scalar T> T blasDot(int count, const T* x, const T* y);
/** @brief 计算范数。@param count 元素数。@param x 输入。@param oneNorm 是否计算一范数。
 * @pre 地址有效且所属当前设备。@return 范数。@code auto value = blasNorm(3, x, false); @endcode
 */
template<Scalar T> T blasNorm(int count, const T* x, bool oneNorm);
/** @brief 转置行主序矩阵。@param rows 行数。@param columns 列数。@param input 输入。@param output 输出。
 * @pre 维度正且地址有效，输入输出不重叠。@return 无返回值。@code blasTranspose(2, 3, a, b); @endcode
 */
template<Scalar T> void blasTranspose(int rows, int columns, const T* input, T* output);
/** @brief 计算行主序矩阵向量积。@param rows 行数。@param columns 列数。@param matrix 矩阵。
 * @param x 向量。@param y 输出。@pre 维度正且地址有效。@return 无返回值。
 * @code blasGemv(2, 3, a, x, y); @endcode
 */
template<Scalar T> void blasGemv(int rows, int columns, const T* matrix, const T* x, T* y);
/** @brief 计算行主序矩阵乘积。@param rows 结果行数。@param columns 结果列数。
 * @param inner 内维度。@param a 左矩阵。@param b 右矩阵。@param c 输出。
 * @pre 维度正且地址有效。@return 无返回值。@code blasGemm(2, 4, 3, a, b, c); @endcode
 */
template<Scalar T> void blasGemm(int rows, int columns, int inner, const T* a, const T* b, T* c);
/** @brief 用 cuSPARSE 计算 CSR 向量积。@param matrix 已规范化的设备 CSR。@param x 输入。@param y 输出。
 * @pre 非零矩阵及指针属于当前设备。@return 无返回值。@code sparseMv(matrix, x, y); @endcode
 */
template<Scalar T, CsrIndex I> void sparseMv(const CsrMatrix<T, I>& matrix, const T* x, T* y);
/** @brief 用 cuSPARSE 计算 CSR 与行主序矩阵乘积。@param matrix 已规范化 CSR。
 * @param columns 右矩阵列数。@param b 右矩阵。@param c 输出。
 * @pre 非零矩阵及指针属于当前设备。@return 无返回值。@code sparseMm(matrix, 3, b, c); @endcode
 */
template<Scalar T, CsrIndex I> void sparseMm(const CsrMatrix<T, I>& matrix, std::size_t columns, const T* b, T* c);
/** @brief 用 cuSPARSE 合并两个 CSR。@param a 左矩阵。@param b 右矩阵。@param beta 右矩阵系数。
 * @pre 形状相同，结构有序且无重复，索引为 int。@return 主机结果。
 * @code auto result = sparseAdd(a, b, 1.0); @endcode
 */
template<Scalar T> SparseHostData<T> sparseAdd(const CsrMatrix<T, int>& a, const CsrMatrix<T, int>& b, T beta);
/** @brief 用 cuSPARSE 转置 CSR。@param matrix 规范化 int 索引矩阵。
 * @pre 非零且属于当前设备。@return 主机结果。@code auto result = sparseTranspose(matrix); @endcode
 */
template<Scalar T> SparseHostData<T> sparseTranspose(const CsrMatrix<T, int>& matrix);
/** @brief 用 cuSPARSE 计算稀疏乘积。@param a 左矩阵。@param b 右矩阵。
 * @pre 内维度匹配，规范化的非零 int 索引矩阵属于当前设备。@return 主机结果。
 * @code auto result = sparseGemm(a, b); @endcode
 */
template<Scalar T> SparseHostData<T> sparseGemm(const CsrMatrix<T, int>& a, const CsrMatrix<T, int>& b);
} // 命名空间 detail
} // 命名空间 thermofem::linalg

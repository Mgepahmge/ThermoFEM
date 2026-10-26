/** @file example-3.cu
 * @brief 展示向量运算符与 cuSPARSE 矩阵向量积。
 */
#include <LinearAlgebra/LinearAlgebra.cuh>
#include <iostream>

/** @brief 执行线性代数示例；无参数、无额外约束。
 * @return 成功返回零。@code ./example-3 @endcode
 */
int main() {
    using namespace thermofem;
    const std::vector<int> offsets{0, 2, 4}, columns{0, 1, 0, 1};
    const std::vector<double> values{4, -1, -1, 4}, input{1, 2};
    const CsrMatrix<double> matrix(2, 2, offsets, columns, values);
    const Vector<double> x{std::span<const double>(input)};
    const auto y = matrix * x + 2 * x;
    std::cout << "norm(A*x + 2*x) = " << linalg::norm2(y) << '\n';
}

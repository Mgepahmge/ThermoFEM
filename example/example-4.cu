/** @file example-4.cu
 * @brief 使用 Jacobi、AMG、BiCGStab 求解二维 Poisson 五点离散系统。
 */
#include <Solver/BiCgStab.cuh>
#include <iostream>

/** @brief 构造已知解为一的二维系统并输出求解报告，无参数。
 * @pre CUDA 设备可用。@return 收敛且误差达标返回零，否则返回一。
 * @code ./example-4 @endcode
 */
int main() {
    using namespace thermofem;
    constexpr int side = 16, n = side * side;
    std::vector<int> offsets{0}, columns;
    std::vector<double> entries, b(n), initial(n, 0);
    for (int row = 0; row < n; ++row) {
        int neighbors{};
        for (int neighbor : {row - side, row - 1, row + 1, row + side}) {
            if (neighbor < 0 || neighbor >= n || (std::abs(neighbor - row) == 1 && neighbor / side != row / side)) {
                continue;
            }
            columns.push_back(neighbor);
            entries.push_back(-1);
            ++neighbors;
        }
        columns.push_back(row);
        entries.push_back(4);
        offsets.push_back(static_cast<int>(entries.size()));
        b[row] = 4 - neighbors;
    }
    const CsrMatrix<double> matrix(n, n, offsets, columns, entries);
    const Vector<double> rhs{std::span<const double>(b)};
    Vector<double> x{std::span<const double>(initial)};
    BiCgStabOptions<double> options;
    options.relativeTolerance_ = 1e-10;
    BiCgStab<double> solver(options);
    solver.setup(matrix);
    const auto report = solver.apply(rhs, x);
    std::vector<double> solution(n);
    x.copyToHost(solution);
    double maximumError{};
    for (double value : solution) {
        maximumError = std::max(maximumError, std::abs(value - 1));
    }
    std::cout << "Iterations: " << report.iterations_ << "\nRelative residual: " << report.relativeResidual_
              << "\nMaximum error: " << maximumError << '\n';
    return report.status_ == SolveStatus::Converged && maximumError < 1e-6 ? 0 : 1;
}

/** @file solve-matrix.cu
 * @brief 读取真实稀疏矩阵，通过构造已知解验证求解器的收敛与前向误差。
 */
#include <IO/MatrixMarket.cuh>
#include <Solver/BiCgStab.cuh>
#include <Preconditioner/Identity.cuh>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

/** @brief 提供测试程序内部的状态显示和独立主机校验。 */
namespace {
/** @brief 将终止状态转换为文本。@param status 状态。@pre 有效枚举。
 * @return 稳定的状态名称。@code statusName(report.status_); @endcode */
const char* statusName(thermofem::SolveStatus status) {
    using thermofem::SolveStatus;
    switch (status) {
    case SolveStatus::Converged: return "Converged";
    case SolveStatus::MaxIterations: return "MaxIterations";
    case SolveStatus::Breakdown: return "Breakdown";
    case SolveStatus::NonFinite: return "NonFinite";
    }
    return "Unknown";
}
/** @brief 独立计算主机矩阵向量积，使用扩展精度累加。
 * @param matrix 主机 CSR。@param x 输入。@pre 长度匹配。
 * @return 双精度结果。@code auto b = hostMultiply(matrix, exact); @endcode */
std::vector<double> hostMultiply(const thermofem::io::HostCsr& matrix, const std::vector<double>& x) {
    std::vector<double> result(static_cast<std::size_t>(matrix.rows_));
    for (int row = 0; row < matrix.rows_; ++row) {
        long double sum{};
        for (int j = matrix.offsets_[row]; j < matrix.offsets_[row+1]; ++j) {
            sum += static_cast<long double>(matrix.values_[j])*x[matrix.indices_[j]];
        }
        result[row] = static_cast<double>(sum);
    }
    return result;
}
}
/** @brief 执行真实矩阵的制造解验证。
 * @param argc 参数数目。@param argv 参数：矩阵路径，可选 amg/identity 和 v/w/f。
 * @pre 可用 CUDA 设备，输入为非空实数方阵。@return 两种校验均通过返回零，数值失败返回二，输入或执行错误返回一。
 * @details 使用确定性的非恒定 x，主机计算 b=A*x，零初值、double、rtol=1e-8、atol=0、
 * 最大 1000 次迭代。主机独立校验残差与已知解相对误差，后者阈值 1e-4。
 * 输出读取、setup、apply 时间；不将文件读取或 CUDA 运行时预热计入迭代时间。
 * 单次运行不构成性能基准；制造解验证不等于原始工况物理解验证。
 * @code ./build/example/solve-matrix data/kvlcc-351k-Velx-1000.mtx amg @endcode
 */
int main(int argc, char** argv) {
    using namespace thermofem;
    using Clock = std::chrono::steady_clock;
    std::cout << std::unitbuf << std::setprecision(12);
    try {
        if (argc < 2 || argc > 4) { throw std::invalid_argument("Usage: solve-matrix path [amg|identity] [v|w|f]"); }
        const std::string mode = argc >= 3 ? argv[2] : "amg";
        const std::string cycle = argc == 4 ? argv[3] : "v";
        if(cycle!="v"&&cycle!="w"&&cycle!="f")throw std::invalid_argument("Unknown cycle");
        if (mode != "amg" && mode != "identity") { throw std::invalid_argument("Unknown preconditioner"); }
        const auto begin = Clock::now();
        std::ifstream file(argv[1]);
        if (!file) { throw std::runtime_error("Cannot open matrix file"); }
        const auto host = io::readMatrixMarket(file);
        if (host.rows_ <= 0 || host.rows_ != host.columns_) { throw std::invalid_argument("Expected nonempty square matrix"); }
        const auto loaded = Clock::now();
        const auto n = static_cast<std::size_t>(host.rows_);
        std::cout << "matrix=" << argv[1] << "\nrows=" << n << "\nnnz=" << host.values_.size()
                  << "\npreconditioner=" << mode << "\nread_seconds=" << std::chrono::duration<double>(loaded-begin).count() << '\n';
        std::vector<double> exact(n), initial(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            exact[i] = 1.0 + 0.25*std::sin(double(i)*0.013) + 0.125*std::cos(double(i)*0.007);
        }
        const auto b = hostMultiply(host, exact);
        int device{};
        auto cudaStatus = cudaGetDevice(&device);
        if (cudaStatus != cudaSuccess) { throw CudaError(cudaStatus, "cudaGetDevice"); }
        cudaDeviceProp properties{};
        cudaStatus = cudaGetDeviceProperties(&properties, device);
        if (cudaStatus != cudaSuccess) { throw CudaError(cudaStatus, "cudaGetDeviceProperties"); }
        int runtimeVersion{};
        cudaStatus = cudaRuntimeGetVersion(&runtimeVersion);
        if (cudaStatus != cudaSuccess) { throw CudaError(cudaStatus, "cudaRuntimeGetVersion"); }
        std::cout << "device=" << device << "\ngpu=" << properties.name << "\ncuda_runtime=" << runtimeVersion
                  << "\nrtol=" << 1e-8 << "\natol=0\nmax_iterations=1000\nrefresh_interval=50\n";
        const CsrMatrix<double> matrix(n, n, host.offsets_, host.indices_, host.values_);
        const Vector<double> rhs{std::span<const double>(b)};
        Vector<double> x{std::span<const double>(initial)};
        AmgOptions<double> amgOptions;
        amgOptions.cycle_=cycle=="v"?multigrid::Cycle::V:(cycle=="w"?multigrid::Cycle::W:multigrid::Cycle::F);
        auto amg = std::make_unique<Amg<double>>(amgOptions);
        auto* hierarchy = amg.get();
        std::unique_ptr<Preconditioner<double>> preconditioner;
        if (mode == "amg") { preconditioner = std::move(amg); }
        else { preconditioner = std::make_unique<Identity<double>>(); }
        BiCgStabOptions<double> options; options.relativeTolerance_ = 1e-8;
        BiCgStab<double> solver(options, std::move(preconditioner));
        std::cout << "phase=setup\n";
        const auto setupBegin = Clock::now(); solver.setup(matrix); const auto setupEnd = Clock::now();
        std::cout << "setup_seconds=" << std::chrono::duration<double>(setupEnd-setupBegin).count() << '\n';
        if (mode == "amg") {
            std::cout << "cycle=" << cycle << "\nlevels=" << hierarchy->levelCount()
                      << "\nstop_reason=" << hierarchy->stopReason() << "\ncoarse_solver=" << hierarchy->coarseSolver() << '\n';
            std::size_t totalRows{},totalNonzeros{};
            for(std::size_t i=0;i<hierarchy->levelStatistics().size();++i){
                const auto& s=hierarchy->levelStatistics()[i];totalRows+=s.rows_;totalNonzeros+=s.nonzeros_;
                std::cout << "level=" << i << " rows=" << s.rows_ << " nnz=" << s.nonzeros_ << " aggregates=";
                for(auto count:s.aggregateSizes_)std::cout << count << ',';
                std::cout << '\n';
            }
            std::cout << "grid_complexity=" << double(totalRows)/n << "\noperator_complexity=" << double(totalNonzeros)/host.values_.size() << '\n';
        }
        std::cout << "phase=solve\n";
        const auto report = solver.apply(rhs, x); const auto solved = Clock::now();
        std::vector<double> actual(n); x.copyToHost(actual);
        const auto product = hostMultiply(host, actual);
        long double residualSquared{}, normBSquared{}, errorSquared{}, exactSquared{};
        bool finite = true;
        for (std::size_t i = 0; i < n; ++i) {
            finite = finite && std::isfinite(actual[i]) && std::isfinite(product[i]);
            const long double residual = static_cast<long double>(b[i])-product[i];
            const long double error = static_cast<long double>(actual[i])-exact[i];
            residualSquared += residual*residual; normBSquared += static_cast<long double>(b[i])*b[i];
            errorSquared += error*error; exactSquared += static_cast<long double>(exact[i])*exact[i];
        }
        const auto hostResidual = std::sqrt(residualSquared/(normBSquared > 0 ? normBSquared : 1));
        const auto relativeError = std::sqrt(errorSquared/exactSquared);
        std::cout << "status=" << statusName(report.status_) << "\niterations=" << report.iterations_
                  << "\nsolve_seconds=" << std::chrono::duration<double>(solved-setupEnd).count()
                  << "\nreported_relative_residual=" << report.relativeResidual_
                  << "\nhost_relative_residual=" << hostResidual << "\nrelative_solution_error=" << relativeError << '\n';
        const bool passed = finite && report.status_ == SolveStatus::Converged && hostResidual <= 1.01e-8 && relativeError <= 1e-4;
        std::cout << "validation=" << (passed ? "PASS" : "FAIL") << '\n';
        return passed ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "error=" << error.what() << '\n'; return 1;
    }
}

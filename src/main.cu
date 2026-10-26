/**
 * @file main.cu
 * @brief 提供 ThermoFEM 命令行入口，当前输出库版本信息。
 */

#include <thermofem/version.cuh>

#include <iostream>

/**
 * @brief 输出当前链接的 ThermoFEM 库版本。
 * @details 无参数。
 * @pre 无额外约束，不要求可用的 CUDA 设备。
 * @return 正常结束时返回 0。
 * @code{.sh}
 * ./build/thermofem
 * @endcode
 */
int main() {
    const auto currentVersion = thermofem::version();
    std::cout << "ThermoFEM " << currentVersion.major_ << '.' << currentVersion.minor_ << '.'
              << currentVersion.patch_ << '\n';
    return 0;
}

/**
 * @file example-1.cu
 * @brief 演示通过公共头文件调用 ThermoFEM 库版本查询接口。
 */

#include <thermofem/version.cuh>

#include <iostream>

/**
 * @brief 查询并输出 ThermoFEM 库版本，演示基础库的链接与调用。
 * @details 无参数。
 * @pre 无额外约束，不启动 CUDA 内核。
 * @return 正常结束时返回 0。
 * @code{.sh}
 * ./build/example/example-1
 * @endcode
 */
int main() {
    const auto currentVersion = thermofem::version();
    std::cout << "Example 1: ThermoFEM library version " << currentVersion.major_ << '.'
              << currentVersion.minor_ << '.' << currentVersion.patch_ << '\n';
    return 0;
}

/**
 * @file version.cu
 * @brief 实现 ThermoFEM 库版本查询接口。
 */

#include <thermofem/version.cuh>

namespace thermofem {

    Version version() noexcept { return {0, 1, 0}; }

} // namespace thermofem

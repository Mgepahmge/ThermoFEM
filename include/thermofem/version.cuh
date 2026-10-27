/**
 * @file version.cuh
 * @brief 定义 ThermoFEM 版本数据结构与版本查询接口。
 */

#pragma once

/** @brief 提供热传导有限元求解器的公共接口及其基础数据类型。 */
namespace thermofem {

    /**
     * @brief 保存语义化版本号的主版本、次版本和修订版本。
     *
     * 此类型仅持有整数值，无需管理外部资源。
     * @code{.cpp}
     * const auto currentVersion = thermofem::version();
     * const int majorVersion = currentVersion.major_;
     * @endcode
     */
    struct Version {
        int major_{}; ///< 主版本号，表示不兼容的接口变更。
        int minor_{}; ///< 次版本号，表示向后兼容的功能扩展。
        int patch_{}; ///< 修订版本号，表示向后兼容的问题修复。
    };

    /**
     * @brief 查询当前链接的 ThermoFEM 库版本。
     * @details 无参数，不抛出异常。
     * @pre 无额外约束，不要求初始化 CUDA 设备。
     * @return 包含主版本号、次版本号和修订版本号的版本对象。
     * @code{.cpp}
     * const auto currentVersion = thermofem::version();
     * const int minorVersion = currentVersion.minor_;
     * @endcode
     */
    [[nodiscard]] Version version() noexcept;

} // namespace thermofem

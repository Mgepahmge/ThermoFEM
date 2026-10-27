/**
 * @file version_test.cpp
 * @brief 验证当前链接的 ThermoFEM 库返回初始版本号。
 */

#include <thermofem/version.cuh>

#include <gtest/gtest.h>

/**
 * @brief 检查版本查询接口返回的三个版本分量。
 * @details 无运行时参数；测试套件与用例名称由 GoogleTest 宏指定。
 * @pre 无额外约束，不要求初始化 CUDA 设备。
 * @note 无返回值，通过 GoogleTest 断言报告结果。
 * @code{.sh}
 * ./build/test/thermofem_tests --gtest_filter=VersionTest.returnsInitialProjectVersion
 * @endcode
 */
TEST(VersionTest, returnsInitialProjectVersion) {
    const auto currentVersion = thermofem::version();
    EXPECT_EQ(currentVersion.major_, 0);
    EXPECT_EQ(currentVersion.minor_, 1);
    EXPECT_EQ(currentVersion.patch_, 0);
}

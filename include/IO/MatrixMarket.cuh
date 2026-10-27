/** @file MatrixMarket.cuh
 * @brief 提供独立的 Matrix Market 实数坐标矩阵读取接口。
 */
#pragma once
#include <istream>
#include <vector>

/** @brief 提供文件格式解析，与 CUDA 存储及求解组件解耦。 */
namespace thermofem::io {
    /** @brief 保存主机端双精度、32 位索引 CSR 数据。
     * @details 输入坐标可乱序或重复，读取后按行整理，重复项保留供后续规范化。
     * @code auto host = readMatrixMarket(stream); @endcode
     */
    struct HostCsr {
        int rows_{}; ///< 行数。
        int columns_{}; ///< 列数。
        std::vector<int> offsets_; ///< CSR 行偏移。
        std::vector<int> indices_; ///< 零基列索引。
        std::vector<double> values_; ///< 实数条目值。
    };
    /** @brief 解析 Matrix Market coordinate real/integer general/symmetric。
     * @param input 输入文本流，不接管其生命周期。
     * @pre 流可读；尺寸、展开后的非零数不超过 INT_MAX，数值有限。
     * @return 主机 CSR，symmetric 自动补全转置的非对角条目。
     * @throws std::runtime_error 格式、条目数、索引、数值或对称存储不合法。
     * @details 拒绝 array、complex、pattern 等未支持格式，避免误解释输入。
     * 使用逐行结构解析，忽略空行及注释；对称输入要求只提供同一侧三角。
     * 空间为 O(rows+nnz)，允许重复项但不合并；不依赖 CUDA，便于独立单元测试。
     * @code std::ifstream file(path); auto host = readMatrixMarket(file); @endcode
     */
    HostCsr readMatrixMarket(std::istream& input);
} // namespace thermofem::io

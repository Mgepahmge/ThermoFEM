/** @file NativeAggregationOracle.cuh
 * @brief 定义测试专用的独立主机聚合基准，不参与 AMG 的 GPU setup。
 */
#pragma once
#include <IO/MatrixMarket.cuh>
#include <array>
#include <cstdint>

namespace thermofem::testoracle {
    /** @brief 主机 CSR 图，权重非负、无自环、每行列有序。
     * @code auto graph = strengthGraph(matrix, 0.25); @endcode
     */
    struct StrengthGraph {
        io::HostCsr edges_; ///< 对称强连接图，反向边取归一化强度的最大值。
    };
    /** @brief 保存聚合映射与按细层编号排序的成员表。
     * @details 两轮匹配后每个聚合最多四个成员。成员表可直接用于无原子加法的限制。
     * @code auto transfer = aggregate(graph, 8, true); @endcode
     */
    struct AggregateMap {
        std::vector<int> fineToCoarse_; ///< 每个细层顶点的粗层编号。
        std::vector<int> offsets_; ///< 各聚合成员的 CSR 偏移。
        std::vector<int> members_; ///< 每个聚合内按编号升序的成员。
        std::array<std::size_t, 4> histogram_{}; ///< 一至四成员聚合的数量。
        /** @brief 查询聚合数量，无参数、无额外约束。@return 粗层维度。
         * @code auto n = transfer.coarseRows(); @endcode */
        int coarseRows() const noexcept { return offsets_.empty() ? 0 : static_cast<int>(offsets_.size() - 1); }
    };
    /** @brief 检查并规范化主机方阵，排序、合并重复项，保留显式零。
     * @param matrix 输入。@pre 有限值非空合法方阵，条目数不超过 INT_MAX。
     * @return 列有序且无重复的主机 CSR。@code auto a = canonicalMatrix(matrix); @endcode
     */
    io::HostCsr canonicalMatrix(const io::HostCsr& matrix);
    /** @brief 建立双向强连接图。@param matrix 规范化方阵。@param threshold 相对行最大非对角阈值。
     * @pre 合法方阵，threshold 在 [0,1]，双向边数可由 int 表示。
     * @return 同一边两方向权重相同、取各方向归一化强度最大值的图。
     * @details 仅非零非对角条目参与，规则为 abs(aij)/rowMaximum>=threshold。
     * 此运算不要求对角非零，层次构造负责对角校验。@code auto g = strengthGraph(a, 0.25); @endcode
     */
    StrengthGraph strengthGraph(const io::HostCsr& matrix, double threshold);
    /** @brief 同步提议并接受相互选择的边，按根编号压缩聚合编号。
     * @param graph 强连接图。@param rounds 匹配轮数，1 至 8。
     * @pre 合法非负权重图。@return 每个顶点的配对聚合编号。
     * @details 权重相同使用与参考实现相同的对称边哈希排序，再按邻点编号破同值。
     * 每轮提议基于同一匹配快照，不能用顺序贪心替代。@code auto map = matchPairs(g, 8); @endcode
     */
    std::vector<int> matchPairs(const StrengthGraph& graph, int rounds);
    /** @brief 按已有映射收缩图，跨聚合边取最大权重，删除聚合内边。
     * @param graph 输入。@param map 聚合编号。@pre 映射长度匹配且编号连续。
     * @return 收缩后的图。@code auto contracted = contractGraph(g, map); @endcode
     */
    StrengthGraph contractGraph(const StrengthGraph& graph, const std::vector<int>& map);
    /** @brief 两轮匹配并合成映射，可选执行两轮单点吸收。
     * @param graph 强连接图。@param rounds 每轮匹配内部的同步提议次数。
     * @param absorbSingletons 是否吸收单点。@pre 输入图合法，rounds 在 [1,8]。
     * @return 最多四成员的聚合映射及成员表。
     * @details 第一轮匹配原图，第二轮匹配收缩图，然后合成细到粗映射。
     * 单点只能进入大小为 2 或 3 的聚合；目标按权重降序、细点编号升序选择容量内成员。
     * 两轮吸收均读取轮前快照，结束后压缩空聚合编号，与参考实现保持同一语义。
     * @code auto map = aggregate(g, 8, true); @endcode
     */
    AggregateMap aggregate(const StrengthGraph& graph, int rounds, bool absorbSingletons);
    /** @brief 以聚合行列归并求和计算 Ac=P^T*A*P，不构造稀疏矩阵乘积。
     * @param matrix 细层方阵。@param transfer 聚合映射。
     * @pre 输入规范化、维度匹配。@return 有序粗层 CSR，保留归并后的零条目。
     * @details P 每行仅有一个值为一的条目。每个细层非零项恰好映射到一个粗层坐标，
     * 所以 coarse.nnz<=fine.nnz；不使用平滑延拓、截断或扩大 SpGEMM 工作区。
     * @code auto coarse = aggregateGalerkin(a, transfer); @endcode
     */
    io::HostCsr aggregateGalerkin(const io::HostCsr& matrix, const AggregateMap& transfer);
} // namespace thermofem::testoracle

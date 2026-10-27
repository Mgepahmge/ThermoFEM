/** @file Cycle.cuh
 * @brief 定义不依赖矩阵、平滑器和 GPU 执行方式的 V/W/F 循环调度。
 */
#pragma once
#include <concepts>
#include <cstddef>
#include <stdexcept>
namespace thermofem::multigrid {
    /** @brief V 单次递归、W 两次递归、F 嵌套迭代后 V 修正。 */
    enum class Cycle { V, W, F };
    /** @brief 约束独立层运算，不要求继承或特定存储。
     * @details begin 记录访问；clear 清零；coarseSolve 使用当前初值；restrictRhs 限制
     * 原右端；restrictResidual 限制当前残差；prolong 累加粗层解。
     * @code static_assert(CycleOperations<MyHierarchy>); @endcode */
    template <class O>
    concept CycleOperations = requires(O& operations, std::size_t level) {
        { operations.isCoarsest(level) } -> std::convertible_to<bool>;
        operations.begin(level);
        operations.clear(level);
        operations.coarseSolve(level);
        operations.preSmooth(level);
        operations.postSmooth(level);
        operations.restrictRhs(level);
        operations.restrictResidual(level);
        operations.prolong(level);
    };
    /** @brief 调度完整多重网格循环，不分配缓存。
     * @param kind 循环类型。@param level 起始层。@param operations 层运算适配器。
     * @param zeroInitial 是否丢弃初值。@pre 层次完整，不可并发访问同一缓存。
     * @note 无返回值。
     * @details V/W 前平滑、限制残差、递归求误差、延拓并后平滑。W 第二次递归保留
     * 第一次结果，右端不变。F 对应参考的嵌套迭代：限制原右端、F 求粗初值、延拓覆盖
     * 本层解，再执行 V 修正。固定循环和平滑次数维持线性，不执行收敛判据。
     * @code executeCycle(Cycle::W,0,operations,true); @endcode */
    template <CycleOperations O>
    void executeCycle(Cycle kind, std::size_t level, O& operations, bool zeroInitial = true) {
        if (kind != Cycle::V && kind != Cycle::W && kind != Cycle::F) {
            throw std::invalid_argument("Unknown AMG cycle");
        }
        operations.begin(level);
        if (zeroInitial || kind == Cycle::F) {
            operations.clear(level);
        }
        if (operations.isCoarsest(level)) {
            operations.coarseSolve(level);
            return;
        }
        if (kind == Cycle::F) {
            operations.restrictRhs(level);
            executeCycle(Cycle::F, level + 1, operations, true);
            operations.prolong(level);
            executeCycle(Cycle::V, level, operations, false);
            return;
        }
        operations.preSmooth(level);
        operations.restrictResidual(level);
        executeCycle(kind, level + 1, operations, true);
        if (kind == Cycle::W) {
            executeCycle(kind, level + 1, operations, false);
        }
        operations.prolong(level);
        operations.postSmooth(level);
    }
} // namespace thermofem::multigrid

# 求解组件

`Smoother<T,I>`、`Preconditioner<T,I>` 和 `Solver<T,I>` 为三个独立的虚接口，
均提供 `setup(matrix)` 与 `apply(rhs, x)`，支持 float/double、32/64 位索引。
平滑器从 x 的初值迭代，预条件子覆盖 x，求解器从初值求解并返回报告。
矩阵必须为非空有限值方阵；各输入位于同一 CUDA 设备。

## 架构与扩展边界

继承仅表达算法角色：Jacobi 是 Smoother，AMG 是 Preconditioner，BiCGStab 是 Solver。
依赖通过组合表达：求解器拥有预条件子，AMG 拥有平滑器，向量拥有 Buffer。
矩阵及向量不从资源管理类继承，避免将存储实现当作数学对象的替代类型。

算法组件负责 setup/apply 的状态管理和步骤编排，通过独立数值运算层完成计算。
GPU 执行、粗化方法和平滑器分别通过各自的接口封装。
新增平滑器实现 Smoother，再通过 AMG 的工厂注入；新增预条件子实现 Preconditioner，
再通过 BiCGStab 的构造函数注入，均不需要派生或改写外层组件。

但接口相容不等于算法相容：本 BiCGStab 依赖固定线性预条件动作。
非线性、按残差自适应或每次改变扫描次数的预条件子需要适配的外层算法。
AMG 的平滑器工厂应返回彼此独立的实例，每层拥有自己的缓存。

setup 保存矩阵快照并分配缓存，改变原矩阵不会影响现有状态。
apply 为同步调用，接受 global/pinned/mapped/managed 向量，保留输出存储策略和地址。
右端项与解允许别名。各实例拥有可变工作区，不支持同一实例的并发 apply。

## Jacobi 平滑器

`<Smoother/Jacobi.cuh>` 中的 `Jacobi<T,I>(sweeps, weight)` 使用加权 Jacobi：
`x <- x + weight * D^-1 * (b-A*x)`。默认两次扫描，权重为 2/3。
setup 合并重复 CSR 元素，检查缺失、零及倒数不有限的对角线。
失败的 setup 保留旧状态；无 setup、尺寸不匹配或跨设备调用抛异常。

`linalg::IterationContext` 复用 cuBLAS/cuSPARSE 句柄、SpMV 描述符和工作区。
GPU Jacobi 内核与库调用使用默认流并同步返回。

## AMG 预条件子

`<Preconditioner/Amg.cuh>` 的 `Amg<T,I>` 使用 GPU 原生聚合构造层次，算法参考
作者的非公开项目 YHAMG-GPU。层次构造分为以下阶段：

1. GPU 检查 CSR 是否规范；必要时按坐标排序、重复项求和，保留显式零。
2. 每层校验所有值有限、对角非零且倒数有限。以行最大非对角绝对值归一化强度，
   达到阈值的边双向插入，重复方向取最大权重，删除自环。
3. 在强图执行最多 8 轮同步提议及互选接受。等权边按参考的对称 64 位哈希破同值，
   再按邻点编号排序。按根编号前缀和产生连续聚合编号。
4. 收缩强图，跨聚合边取最大权重，再做一阶段同步匹配，合成细到粗映射。
5. 按需进行两轮单点吸收，每轮使用快照。单点只向二、三成员聚合提议，目标按
   强度降序、细点编号升序接收，容量最多四。最后压缩空编号、排序成员并统计。
6. 每个细点的延拓只有一个值为一的条目，`P[i,map[i]]=1`，`R=P^T`。
   将细矩阵条目映射为 `(map[row],map[column],value)`，GPU 排序求和得到粗矩阵。
   每个细条目只映射一次，因此每层 `nnz(Ac)<=nnz(A)`；不构造 SpGEMM 中间矩阵。
7. 达到目标粗阶数、最大层数、空强图或无法收缩时停止。<=64 阶时在 GPU 上执行
   部分主元 LU，主元阈值为 `64*epsilon(double)*max(abs(A))`，缓存因子与交换表。
   奇异/非有限因子或终止层大于 64 阶时，采用固定次数加权 Jacobi。

默认配置：最大 20 层，粗阶数目标 64，两阶段各 8 轮匹配，启用两轮单点吸收，
前后各一次 Jacobi、权重 0.75，粗层 fallback 为 8 次 Jacobi，强度阈值 0.25。
`compactSortKeys_` 仅减少排序键位宽，不改变聚合算法。原生 setup 使用 double/int；
float 与 64 位索引在 GPU 边界转换，迭代计算保持原类型，原生图仍受 int32 容量限制。

`AmgOptions::cycle_` 在以下三种调度中选择；实现位于独立的 `<Multigrid/Cycle.cuh>`，
通过受约束的运算适配器组合每层操作，不依赖 Amg、矩阵类型或 CUDA：

| 循环 | 递归与初值语义 |
| --- | --- |
| V | 前平滑，限制残差，粗层零初值 V，延拓累加，后平滑 |
| W | 同 V，但对同一粗右端连续执行两次 W；第二次保留第一次粗解 |
| F | 限制原右端，递归 F 得到粗初值，延拓覆盖本层解，再从此初值执行一次 V |

F 使用参考代码的嵌套迭代定义，不能用两个 V 代替。W 中第二次递归若清零则丢失
第一次校正。最粗层若用 LU 则覆盖解；若用 Jacobi 则继续已有初值。apply 每次从零
开始，固定次数维持线性，不自动判断收敛。前后平滑次数可以独立为零。

```cpp
thermofem::AmgOptions<double> amgOptions;
amgOptions.cycle_ = thermofem::multigrid::Cycle::F;
amgOptions.preSweeps_ = 1;
amgOptions.postSweeps_ = 1;
auto preconditioner = std::make_unique<thermofem::Amg<double>>(amgOptions);
thermofem::BiCgStab<double> solver({}, std::move(preconditioner));
```

工厂分别为每层前后平滑建立独立实例，避免共享可变缓存；注入工厂时扫描数由其
返回组件定义，配置为零时跳过对应阶段。粗层 fallback 使用独立 Jacobi。
`levelStatistics()`、`stopReason()`、`coarseSolver()` 与 `lastCycleVisits()` 用于诊断。
setup 以完整临时状态构建，失败保留旧状态。一般非对称、不定或各向异性矩阵不保证收敛。

循环通过独立接口依次执行平滑、残差、限制和延拓，各运算同步返回。
矩阵上下文的规范化和逆对角提取包含主机往返，使用时应将 setup 成本与 apply 成本分别考虑。

## BiCGStab 求解器

`<Solver/BiCgStab.cuh>` 的 `BiCgStab<T,I>` 实现右预条件 BiCGStab，默认使用 AMG。
可通过构造函数传入任意 `std::unique_ptr<Preconditioner<T,I>>`；
`<Preconditioner/Identity.cuh>` 提供恒等预条件子，适用于无预条件基线验证。
本项目不使用 cuSOLVER。

```cpp
thermofem::BiCgStabOptions<double> options;
options.relativeTolerance_ = 1e-10;
thermofem::BiCgStab<double> solver(options);
solver.setup(matrix);
auto report = solver.apply(rhs, x);
```

停止条件为 `||b-A*x||2 <= max(atol, rtol*||b||2)`。默认 float 相对容差 1e-5，
double 为 1e-8；较大或病态问题需根据精度和条件数选择可达到的容差。
零右端项时可配置绝对容差；初值已满足条件时报告零次迭代。
所有正常返回路径都重新计算真实残差，防止只凭递推残差误报收敛。
默认每 50 次迭代刷新真实残差并重启递推，候选收敛时也会校验并按需重启。

`SolveResult` 提供 `status_`、`iterations_`、`residualNorm_`、`relativeResidual_`。
状态分别为 `Converged`、`MaxIterations`、`Breakdown`、`NonFinite`。
非收敛状态仍返回当前近似解；无 setup、非法尺寸及库调用错误通过异常报告。
零分母或零正交乘积触发 breakdown；非有限标量触发 NonFinite。
零右端项的 relativeResidual 等于绝对残差。

BiCGStab 的 setup 失败会清空就绪状态，因为注入预条件子的内部异常保证可能不同；
重新成功 setup 后可以继续使用。Jacobi、AMG 的 setup 失败仍保留旧状态。
完整二维 Poisson 示例见 `example/example-4.cu`。

## CG 求解器

`<Solver/Cg.cuh>` 提供 `Cg<T,I>` 和 `CgOptions<T>`。默认组合恒等预条件子，
对应普通 CG；也可注入 `std::unique_ptr<Preconditioner<T,I>>` 实现 PCG。
矩阵必须对称正定，预条件动作必须固定、线性且对称正定。
恒等预条件子满足 CG 的数学要求；现有 AMG 与 PCG 的相容性尚未保证。

```cpp
#include <Solver/Cg.cuh>

thermofem::CgOptions<double> options;
options.relativeTolerance_ = 1e-10;
thermofem::Cg<double> solver(options);
solver.setup(matrix);
auto report = solver.apply(rhs, x);
```

默认最多迭代 1000 次，float 相对容差为 1e-5，double 为 1e-8，绝对容差为零。
收敛判据、报告、别名处理及输出内存策略与 BiCGStab 一致；setup 失败使求解器失效。
每次返回均重算真实残差，初值收敛时报告零次迭代，迭代上限为零时仅检查初值。
零右端项通常需要合理的绝对容差，其 relativeResidual 等于绝对残差。

默认每 50 次迭代以及递推残差达到阈值时重新计算真实残差；未收敛则重启搜索方向。
过于频繁的刷新会削弱共轭方向的收益，应结合精度和问题条件数选择周期。
非正的 `r^T*z` 或 `p^T*A*p` 报告 Breakdown，非有限标量报告 NonFinite。
这些检查不等价于对称性或正定性认证：setup 不进行全矩阵认证，调用方必须保证数学条件。
所有向量和稀疏运算复用独立的 IterationContext，不使用 cuSOLVER。

## 独立运算接口

求解组件通过独立运算组合算法：

- `linalg::inverseDiagonal`：CSR 逆对角线提取与校验。
- `linalg::IterationContext`：缓存 cuBLAS/cuSPARSE 资源，提供 multiply、residual、clear、axpy、scale、dot、norm、jacobi。
- `multigrid::GpuAggregation`：GPU 规范化、强图、完整两阶段聚合和直接 Galerkin 归并。
- `multigrid::restrictAggregates/prolongAggregates`：按成员求和限制和映射延拓累加。
- `multigrid::factorCoarse/solveCoarse`：GPU 小规模主元 LU 及三角求解。
- `multigrid::executeCycle`：受约束且不依赖 GPU 的 V/W/F 调度。
- `linalg::CoarseLu`：仍可独立使用的 CPU 主元 LU，不用于 AMG 的粗层路径。

这些运算接口可在平滑器、预条件子和求解器之外独立调用。

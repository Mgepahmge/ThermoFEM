# 求解组件

`Smoother<T,I>`、`Preconditioner<T,I>` 和 `Solver<T,I>` 为三个独立的虚接口，
均提供 `setup(matrix)` 与 `apply(rhs, x)`，支持 float/double、32/64 位索引。
平滑器从 x 的初值迭代，预条件子覆盖 x，求解器从初值求解并返回报告。
矩阵必须为非空有限值方阵；各输入位于同一 CUDA 设备。

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

`<Preconditioner/Amg.cuh>` 的 `Amg<T,I>` 实现平滑聚合 AMG：按行强连接贪心配对，
以分片常数延拓 P0 构造 `P=(I-weight*D^-1*A)*P0`，`R=P^T`，`Ac=R*A*P`。
每次 apply 从零执行一个 V-cycle，前、后平滑通过 `Smoother` 虚接口调用。
构造时可传入平滑器工厂，每层获得独立的平滑器实例。

聚合和延拓构造目前在 CPU 上执行，Galerkin 稀疏乘积使用 cuSPARSE。
最粗层使用自行实现的部分主元 LU，在 setup 缓存因子；apply 下载粗层右端项、
执行三角求解并上传修正。默认粗层目标维度为 16，稠密粗层硬上限为 256。
粗化未达到硬上限、粗层奇异或出现非有限因子时 setup 抛异常并保留旧状态。
该实现主要针对椭圆型矩阵；非对称、不定或强各向异性问题不保证收敛。
它是可验证的初版，尚未移植参考实现的 GPU 配对、融合内核和 CUDA Graph 优化。

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

## 独立运算接口

求解组件通过独立运算组合算法：

- `linalg::inverseDiagonal`：CSR 逆对角线提取与校验。
- `linalg::IterationContext`：缓存 cuBLAS/cuSPARSE 资源，提供 multiply、residual、clear、axpy、scale、dot、norm、jacobi。
- `multigrid::prolongation`：强连接聚合和平滑延拓构造。
- `multigrid::galerkin`：Galerkin 粗化。
- `linalg::CoarseLu`：独立的带主元粗层分解与缓存求解。

这些接口具有不依赖平滑器、预条件子或求解器的独立数学测试。

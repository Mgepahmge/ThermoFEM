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

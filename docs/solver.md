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

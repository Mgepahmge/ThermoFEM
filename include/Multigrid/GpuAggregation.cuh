/** @file GpuAggregation.cuh
 * @brief 提供 YHAMG-GPU 原生聚合算法的独立 GPU 运算及聚合传输接口。
 */
#pragma once
#include <LinearAlgebra/Operations.cuh>
#include <array>
#include <memory>
namespace thermofem::multigrid {
    /** @brief 组合拥有分片常数延拓的映射与排序成员表。
     * @details 每个细点恰好属于一个聚合，每聚合最多四点。P[i,map[i]]=1，R=P^T。
     * offsets/members 按细点编号排序，限制逐聚合求和，不使用浮点原子操作。
     * @code auto transfer = setup.aggregate(graph,8,true); @endcode */
    struct AggregateTransfer {
        CudaBuffer<int> aggregate_, offsets_, members_; ///< 映射、成员偏移、排序成员。
        int fineRows_{}, coarseRows_{}; ///< 两层维度。
        std::array<int, 4> histogram_{}; ///< 一到四点聚合数。
    };
    /** @brief GPU 小规模部分主元 LU 的所有者，失败时保留状态供调用方选择 Jacobi。
     * @code auto factor = factorCoarse(matrix); @endcode */
    struct CoarseFactor {
        CudaBuffer<double> lu_; ///< 最多 64 阶稠密 LU，以 double 累积。
        CudaBuffer<int> pivots_; ///< 逐步行交换主元。
        bool factored_{false}, singular_{false}; ///< 成功与奇异标志。
        int rows_{}; ///< 因子维度。
    };
    /** @brief 复用 CUB 排序、归并、扫描工作区的独立聚合运算集。
     * @details 不持有层次或循环，不调用平滑器。强图、匹配、收缩、吸收和粗矩阵全部在 GPU 上执行。
     * 原生图索引为 int，权重与归并为 double，与参考实现一致。更宽索引或 float 的转换在组件边界进行。
     * 每次运算同步返回，调用方需先选择矩阵所属设备；实例不可并发调用。
     * @par 算法
     * 强图双向最大归并；固定轮数的同步互选匹配；收缩后再次匹配；可选两轮单点吸收；
     * 每聚合最多四点；Ac 直接映射细层坐标并求和，非零数不增加。不平滑或截断延拓。
     * @code GpuAggregation setup; auto a=setup.canonical(matrix); auto g=setup.strength(a,.25); @endcode */
    class GpuAggregation {
    public:
        /** @brief 分配复用工作区。@param compactKeys 使用最少位数的坐标键。
         * @pre 当前设备有效。@note 无返回值。@code GpuAggregation setup(true); @endcode */
        explicit GpuAggregation(bool compactKeys = false);
        /** @brief 释放工作区，无参数。@pre 无并发访问。@note 无返回值。
         * @code { GpuAggregation setup; } @endcode */
        ~GpuAggregation();
        GpuAggregation(const GpuAggregation&) = delete; ///< 工作区不可复制。
        GpuAggregation& operator=(const GpuAggregation&) = delete; ///< 工作区不可复制。
        /** @brief GPU 排序并求和重复条目；已规范时仅复制。
         * @param a 输入方阵。@pre 非空、有限、合法，rows<INT_MAX，nnz<=INT_MAX/2。
         * @return 规范矩阵。@code auto canonicalA=setup.canonical(a); @endcode */
        CsrMatrix<double> canonical(const CsrMatrix<double>& a);
        /** @brief 校验有限值、非零对角及有限倒数，并缓存行最大非对角强度。
         * @param a 规范方阵。@pre 当前设备正确。@note 无返回值；不合法抛异常。
         * @code setup.validate(a); @endcode */
        void validate(const CsrMatrix<double>& a);
        /** @brief 生成对称强图，各方向的归一化强度取最大值。
         * @param a 规范方阵。@param threshold 相对行最大非对角阈值。
         * @pre threshold 在 [0,1]。@return 有序无自环强图。@code auto g=setup.strength(a,.25); @endcode */
        CsrMatrix<double> strength(const CsrMatrix<double>& a, double threshold);
        /** @brief 原图匹配、收缩匹配、组合映射和两轮单点吸收。
         * @param graph 对称正权无自环强图。@param rounds 匹配轮数 1..8。
         * @param absorbSingletons 是否执行单点吸收。@pre 图由 strength 生成。
         * @return GPU 传输表和聚合统计。@code auto t=setup.aggregate(g,8,true); @endcode */
        AggregateTransfer aggregate(const CsrMatrix<double>& graph, int rounds, bool absorbSingletons);
        /** @brief 映射细矩阵行列后排序求和，计算 Ac=P^T*A*P。
         * @param a 细层方阵。@param t 传输映射。@pre 维度设备匹配。
         * @return 有序粗矩阵，nnz<=a.nnz。@code auto ac=setup.galerkin(a,t); @endcode */
        CsrMatrix<double> galerkin(const CsrMatrix<double>& a, const AggregateTransfer& t);

    private:
        struct Impl; ///< CUB 工作区实现。
        std::unique_ptr<Impl> impl_; ///< 独占工作区。
    };
    /** @brief GPU 聚合限制 coarse=P^T*fine。
     * @param t 映射。@param fine 细层输入。@param coarse 粗层输出。
     * @pre 同设备 global 向量，维度匹配且不别名。@note 无返回值。
     * @code restrictAggregates(t,r,bc); @endcode */
    template <linalg::Scalar T>
    void restrictAggregates(const AggregateTransfer& t, const Vector<T>& fine, Vector<T>& coarse);
    /** @brief GPU 延拓并累加 fine+=P*coarse。
     * @param t 映射。@param coarse 粗层输入。@param fine 细层输入输出。
     * @pre 同设备 global 向量，维度匹配且不别名。@note 无返回值。
     * @code prolongAggregates(t,ec,x); @endcode */
    template <linalg::Scalar T>
    void prolongAggregates(const AggregateTransfer& t, const Vector<T>& coarse, Vector<T>& fine);
    /** @brief GPU 部分主元 LU，阈值为 64*epsilon(double)*max(abs(A))。
     * @param a 规范方阵。@pre 非空且阶数 <=64。@return 因子；奇异或非有限因子标记为失败。
     * @code auto coarse=factorCoarse(a); @endcode */
    CoarseFactor factorCoarse(const CsrMatrix<double>& a);
    /** @brief GPU 主元置换及三角求解，覆盖解。
     * @param factor 因子。@param rhs 右端。@param x 输出。
     * @pre 因子成功，维度匹配，同设备 global；rhs 可与 x 别名。
     * @note 无返回值。@code solveCoarse(factor,b,x); @endcode */
    template <linalg::Scalar T>
    void solveCoarse(const CoarseFactor& factor, const Vector<T>& rhs, Vector<T>& x);
    /** @brief 在 GPU 上转换矩阵数值与索引类型，不排序、不改变结构。
     * @param a 原生 double/int 方阵。@pre 当前设备正确，维度可由 I 表示。
     * @return 转换后的 global 矩阵。@code auto typed=fromNativeMatrix<float,int>(a); @endcode */
    template <linalg::Scalar T, CsrIndex I>
    CsrMatrix<T, I> fromNativeMatrix(const CsrMatrix<double>& a);
    /** @brief 在 GPU 上转换为参考算法的 double/int 存储，检查索引容量。
     * @param a 输入方阵。@pre global，当前设备正确，rows<INT_MAX、nnz<=INT_MAX/2。
     * @return 原生矩阵。@code auto native=toNativeMatrix(a); @endcode */
    template <linalg::Scalar T, CsrIndex I>
    CsrMatrix<double> toNativeMatrix(const CsrMatrix<T, I>& a);
} // namespace thermofem::multigrid

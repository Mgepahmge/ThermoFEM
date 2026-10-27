/** @file Identity.cuh
 * @brief 提供无预条件变换的独立预条件子，便于组件替换与算法验证。
 */
#pragma once
#include <Solver/Interfaces.cuh>
#include <cmath>
namespace thermofem {
    /** @brief 恒等预条件子，以同一接口支持无预条件求解。
     * @code Identity<double> identity; identity.setup(a); identity.apply(b, z); @endcode
     */
    template <linalg::Scalar T, CsrIndex I = int>
    class Identity final : public Preconditioner<T, I> {
    public:
        /** @brief 校验矩阵并记录尺寸设备。@param matrix 输入。
         * @pre 非空有限值方阵。@return 无返回值。@code identity.setup(a); @endcode
         */
        void setup(const CsrMatrix<T, I>& matrix) override {
            iteration::validateMatrix(matrix);
            for (const T value : linalg::detail::hostCopy(matrix.values())) {
                if (!std::isfinite(value)) {
                    throw std::invalid_argument("Nonfinite identity setup matrix");
                }
            }
            size_ = matrix.rowCount();
            device_ = matrix.rowOffsets().deviceId();
        }
        /** @brief 复制右端项，保留输出地址与策略。@param rhs 输入。@param x 输出。
         * @pre setup 成功，尺寸设备匹配。@return 无返回值。@code identity.apply(b, z); @endcode
         */
        void apply(const Vector<T>& rhs, Vector<T>& x) override {
            if (!size_) {
                throw std::logic_error("Identity requires setup");
            }
            iteration::validateVectors(size_, device_, rhs, x);
            x.copyDataFrom(rhs);
        }

    private:
        std::size_t size_{}; ///< 已校验方阵的维度，零表示未初始化。
        int device_{-1}; ///< 所属设备。
    };
} // namespace thermofem

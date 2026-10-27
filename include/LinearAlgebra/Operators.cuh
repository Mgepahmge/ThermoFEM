/** @file Operators.cuh
 * @brief 为向量、稠密矩阵和 CSR 矩阵提供复用线性代数封装的运算符。
 */
#pragma once
#include <LinearAlgebra/Operations.cuh>

namespace thermofem {
    /// 限定为具有受支持浮点类型和线性代数缩放接口的数据结构。
    template <class T>
    concept LinearObject = requires(const T& value) {
        typename T::ValueType;
        requires linalg::Scalar<typename T::ValueType>;
        { linalg::scale(value, typename T::ValueType{1}) } -> std::same_as<T>;
    };

    /** @brief 计算同类型对象之和。@tparam Object 数据结构类型。@param left 左值。@param right 右值。
     * @pre 形状和设备匹配，满足 linalg::add 的限制。@return 沿用左策略的新对象。
     * @code auto result = a + b; @endcode
     */
    template <LinearObject Object>
    Object operator+(const Object& left, const Object& right) {
        return linalg::add(left, right);
    }
    /** @brief 计算同类型对象之差。@tparam Object 数据结构类型。@param left 左值。@param right 右值。
     * @pre 同 linalg::subtract。@return 沿用左策略的新对象。@code auto result = a - b; @endcode
     */
    template <LinearObject Object>
    Object operator-(const Object& left, const Object& right) {
        return linalg::subtract(left, right);
    }
    /** @brief 复制一元正号对象。@tparam Object 数据结构类型。@param value 输入。
     * @pre 外部访问已同步。@return 深拷贝。@code auto copy = +a; @endcode
     */
    template <LinearObject Object>
    Object operator+(const Object& value) {
        return value;
    }
    /** @brief 逐项取负。@tparam Object 数据结构类型。@param value 输入。
     * @pre 同 linalg::scale。@return 沿用输入策略的新对象。@code auto result = -a; @endcode
     */
    template <LinearObject Object>
    Object operator-(const Object& value) {
        return linalg::scale(value, typename Object::ValueType{-1});
    }
    /** @brief 右乘标量。@tparam Object 数据结构类型。@param value 输入。@param factor 系数。
     * @pre 同 linalg::scale。@return 缩放结果。@code auto result = a * 2; @endcode
     */
    template <LinearObject Object>
    Object operator*(const Object& value, typename Object::ValueType factor) {
        return linalg::scale(value, factor);
    }
    /** @brief 左乘标量。@tparam Object 数据结构类型。@param factor 系数。@param value 输入。
     * @pre 同 linalg::scale。@return 缩放结果。@code auto result = 2 * a; @endcode
     */
    template <LinearObject Object>
    Object operator*(typename Object::ValueType factor, const Object& value) {
        return linalg::scale(value, factor);
    }
    /** @brief 除以标量。@tparam Object 数据结构类型。@param value 输入。@param divisor 除数。
     * @pre 除数非零，否则抛 domain_error，其余同 linalg::scale。@return 缩放结果。
     * @code auto result = a / 2; @endcode
     */
    template <LinearObject Object>
    Object operator/(const Object& value, typename Object::ValueType divisor) {
        if (divisor == typename Object::ValueType{}) {
            throw std::domain_error("Division by zero");
        }
        return linalg::scale(value, typename Object::ValueType{1} / divisor);
    }
    /** @brief 计算支持的代数乘积，向量乘向量为内积，矩阵乘法非逐项乘法。
     * @tparam Left 左类型。@tparam Right 右类型。@param left 左对象。@param right 右对象。
     * @pre 存在 linalg::multiply 重载且满足其维度和设备限制。@return 标量、向量或矩阵。
     * @code auto product = matrix * vector; @endcode
     */
    template <LinearObject Left, LinearObject Right>
        requires requires(const Left& left, const Right& right) { linalg::multiply(left, right); }
    auto operator*(const Left& left, const Right& right) {
        return linalg::multiply(left, right);
    }
    /** @brief 相加后覆盖左对象。@tparam Object 数据结构类型。@param left 左对象。@param right 右对象。
     * @pre 同二元加法。@return 左对象引用，保留左策略，底层地址可能改变。
     * @code a += b; @endcode
     */
    template <LinearObject Object>
    Object& operator+=(Object& left, const Object& right) {
        left = left + right;
        return left;
    }
    /** @brief 相减后覆盖左对象。@tparam Object 数据结构类型。@param left 左对象。@param right 右对象。
     * @pre 同二元减法。@return 左对象引用。@code a -= b; @endcode
     */
    template <LinearObject Object>
    Object& operator-=(Object& left, const Object& right) {
        left = left - right;
        return left;
    }
    /** @brief 缩放并覆盖对象。@tparam Object 数据结构类型。@param value 输入输出。@param factor 系数。
     * @pre 同标量乘法。@return 对象引用。@code a *= 2; @endcode
     */
    template <LinearObject Object>
    Object& operator*=(Object& value, typename Object::ValueType factor) {
        value = value * factor;
        return value;
    }
    /** @brief 除以标量并覆盖对象。@tparam Object 数据结构类型。@param value 输入输出。@param divisor 除数。
     * @pre 除数非零，其余同标量除法。@return 对象引用。@code a /= 2; @endcode
     */
    template <LinearObject Object>
    Object& operator/=(Object& value, typename Object::ValueType divisor) {
        value = value / divisor;
        return value;
    }
    /** @brief 矩阵乘法后覆盖左矩阵。@tparam Object 矩阵类型。@param left 左矩阵。@param right 右矩阵。
     * @pre 乘积类型与左对象一致，内维度匹配。@return 左对象引用，结果可改变形状。
     * @code a *= b; @endcode
     */
    template <LinearObject Object>
        requires requires(const Object& left, const Object& right) {
            { linalg::multiply(left, right) } -> std::same_as<Object>;
        }
    Object& operator*=(Object& left, const Object& right) {
        left = left * right;
        return left;
    }
} // namespace thermofem

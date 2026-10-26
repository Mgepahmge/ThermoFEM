/** @file Vector.cuh
 * @brief 定义复用 CUDA 缓冲区所有权及复制语义的一维向量。
 */
#pragma once

#include <DataStructure/CudaBuffer.cuh>

namespace thermofem {
/** @brief 以连续存储表示一维向量，继承 CudaBuffer 的全部存储操作。
 * @tparam T 满足 BufferElement 的元素类型。
 * @details 向量长度等于底层元素数，无额外资源和维度状态；此处不定义向量运算。
 * @code
 * thermofem::Vector<double> vector(10, thermofem::GlobalMemory{});
 * vector.changeMemoryType(thermofem::PinnedMemory{});
 * @endcode
 */
template<BufferElement T>
class Vector : public CudaBuffer<T> {
public:
    using ValueType = T; ///< 向量元素类型。
    using CudaBuffer<T>::CudaBuffer;
};
} // 命名空间 thermofem

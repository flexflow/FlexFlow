#ifndef _FLEXFLOW_UTILS_CUDA_HELPER_H
#define _FLEXFLOW_UTILS_CUDA_HELPER_H

// #include "flexflow/model.h"
#include "op-attrs/datatype.h"
#include "kernels/accessor.h"
#include "kernels/cuda_helper.h"
#include "kernels/device.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cassert>
#include <cstdio>

namespace FlexFlow {
cudaError_t get_legion_stream(cudaStream_t *stream);


} // namespace FlexFlow

template <typename T>
__global__ void apply_add_with_scale(T *data_ptr, T const *grad_ptr, size_t size, T scale);
#endif // FLEXFLOW_CUDA_KERNELS_H

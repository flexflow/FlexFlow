#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CONCAT_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CONCAT_KERNELS_GPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"

namespace FlexFlow {

void concat_gpu_forward_kernel(
    ffStream_t stream,
    GenericTensorAccessorW const &output,
    std::vector<GenericTensorAccessorR> const &inputs,
    ff_dim_t axis);

void concat_gpu_backward_kernel(
    ffStream_t stream,
    GenericTensorAccessorR const &output_grad,
    std::vector<GenericTensorAccessorW> const &input_grads,
    ff_dim_t axis);

} // namespace FlexFlow

#endif

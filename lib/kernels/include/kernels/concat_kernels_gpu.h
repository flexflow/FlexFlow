#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CONCAT_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CONCAT_KERNELS_GPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "op-attrs/ops/concat_attrs.dtg.h"

namespace FlexFlow {

void concat_gpu_forward_kernel(
    ffStream_t stream,
    ConcatAttrs const &attrs,
    std::vector<GenericTensorAccessorR> const &inputs,
    GenericTensorAccessorW const &output);

void concat_gpu_backward_kernel(
    ffStream_t stream,
    ConcatAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    std::vector<GenericTensorAccessorR> const &inputs,
    std::vector<GenericTensorAccessorW> const &input_grads);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_SPLIT_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_SPLIT_KERNELS_GPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "op-attrs/ops/split_attrs.dtg.h"

namespace FlexFlow {

void split_gpu_forward_kernel(
    ffStream_t stream,
    SplitAttrs const &attrs,
    GenericTensorAccessorR const &input,
    std::vector<GenericTensorAccessorW> const &outputs);

void split_gpu_backward_kernel(
    ffStream_t stream,
    SplitAttrs const &attrs,
    std::vector<GenericTensorAccessorR> const &outputs,
    std::vector<GenericTensorAccessorR> const &output_grads,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad);

} // namespace FlexFlow

#endif

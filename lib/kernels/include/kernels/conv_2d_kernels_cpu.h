#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CONV_2D_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CONV_2D_KERNELS_CPU_H

#include "kernels/accessor.h"
#include "op-attrs/ops/conv_2d_attrs.dtg.h"

namespace FlexFlow {

void conv_2d_cpu_forward_kernel(
    Conv2DAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorR const &filter,
    std::optional<GenericTensorAccessorR> const &bias,
    GenericTensorAccessorW const &output);

void conv_2d_cpu_backward_kernel(
    Conv2DAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &filter,
    GenericTensorAccessorW const &filter_grad,
    std::optional<GenericTensorAccessorW> const &bias_grad);

} // namespace FlexFlow

#endif

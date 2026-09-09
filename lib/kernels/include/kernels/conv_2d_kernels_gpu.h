#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CONV_2D_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CONV_2D_KERNELS_GPU_H

#include "kernels/accessor.h"
#include "kernels/conv_2d_per_device_state.dtg.h"
#include "kernels/device.h"
#include "op-attrs/ops/conv_2d_attrs.dtg.h"

namespace FlexFlow {

Conv2DPerDeviceState conv_2d_gpu_init_kernel(PerDeviceFFHandle const &handle,
                                             Conv2DAttrs const &attrs,
                                             TensorShape const &input_shape,
                                             TensorShape const &output_shape);

void conv_2d_gpu_forward_kernel(
    ffStream_t stream,
    PerDeviceFFHandle const &handle,
    Conv2DPerDeviceState const &per_device_state,
    Conv2DAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorR const &filter,
    std::optional<GenericTensorAccessorR> const &bias,
    GenericTensorAccessorW const &output);

void conv_2d_gpu_backward_kernel(
    ffStream_t stream,
    PerDeviceFFHandle const &handle,
    Conv2DPerDeviceState const &per_device_state,
    Conv2DAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &filter,
    GenericTensorAccessorW const &filter_grad,
    std::optional<GenericTensorAccessorW> const &bias_grad);

void conv_2d_gpu_cleanup_kernel(Conv2DPerDeviceState &per_device_state);

} // namespace FlexFlow

#endif

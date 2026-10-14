#ifndef _FLEXFLOW_OPS_KERNELS_CONV_2D_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_CONV_2D_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/conv_2d_per_device_state.dtg.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/ff_handle.h"
#include "op-attrs/activation.dtg.h"
#include "op-attrs/ops/conv_2d_attrs.dtg.h"

namespace FlexFlow {

std::optional<Conv2DPerDeviceState>
    conv2d_init_kernel(DeviceType device_type,
                       device_handle_t const &handle,
                       std::optional<Activation> activation,
                       int kernel_h,
                       int kernel_w,
                       int groups,
                       int padding_h,
                       int padding_w,
                       int stride_h,
                       int stride_w,
                       GenericTensorAccessorW const &input,
                       GenericTensorAccessorW const &output,
                       float const *filter_ptr,
                       float *filter_grad_ptr);

void conv2d_forward_kernel(
    device_stream_t const &stream,
    std::optional<Conv2DPerDeviceState> const &per_device_state,
    Conv2DAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &output,
    GenericTensorAccessorR const &filter,
    std::optional<GenericTensorAccessorR> const &bias);

void conv2d_backward_kernel(
    device_stream_t const &stream,
    std::optional<Conv2DPerDeviceState> const &per_device_state,
    Conv2DAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &filter,
    GenericTensorAccessorW const &filter_grad,
    std::optional<GenericTensorAccessorR> const &bias,
    std::optional<GenericTensorAccessorW> const &bias_grad);

void conv2d_cleanup_kernel(
    DeviceType device_type,
    std::optional<Conv2DPerDeviceState> &per_device_state);

} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_CONV_2D_KERNELS_H

#ifndef _FLEXFLOW_OPS_KERNELS_SOFTMAX_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_SOFTMAX_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/softmax_per_device_state.dtg.h"
#include "op-attrs/ops/softmax_attrs.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "pcg/device_type.dtg.h"

namespace FlexFlow {

std::optional<SoftmaxPerDeviceState>
    softmax_init_kernel(DeviceType device_type,
                        SoftmaxAttrs const &attrs,
                        TensorShape const &input_shape,
                        TensorShape const &output_shape);

void softmax_forward_kernel(
    device_stream_t const &stream,
    device_handle_t const &handle,
    std::optional<SoftmaxPerDeviceState> const &per_device_state,
    SoftmaxAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &output);

void softmax_backward_kernel(
    device_stream_t const &stream,
    device_handle_t const &handle,
    std::optional<SoftmaxPerDeviceState> const &per_device_state,
    SoftmaxAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad);

void softmax_cleanup_kernel(
    DeviceType device_type,
    std::optional<SoftmaxPerDeviceState> &per_device_state);

} // namespace FlexFlow

#endif

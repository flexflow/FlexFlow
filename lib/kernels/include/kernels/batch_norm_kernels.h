#ifndef _FLEXFLOW_KERNELS_BATCH_NORM_KERNELS_H
#define _FLEXFLOW_KERNELS_BATCH_NORM_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "kernels/batch_norm_per_device_state.dtg.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_stream_t.dtg.h"
#include "op-attrs/ops/batch_norm_attrs.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "pcg/device_type.dtg.h"

namespace FlexFlow {

std::optional<BatchNormPerDeviceState>
    batch_norm_init_kernel(DeviceType device_type,
                           Allocator &allocator,
                           BatchNormAttrs const &attrs,
                           TensorShape const &input_shape,
                           TensorShape const &output_shape);

void batch_norm_forward_kernel(
    device_stream_t const &stream,
    device_handle_t const &handle,
    std::optional<BatchNormPerDeviceState> const &per_device_state,
    BatchNormAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorR const &gamma,
    GenericTensorAccessorR const &beta,
    GenericTensorAccessorW const &output);

void batch_norm_backward_kernel(
    device_stream_t const &stream,
    device_handle_t const &handle,
    std::optional<BatchNormPerDeviceState> const &per_device_state,
    BatchNormAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &gamma,
    GenericTensorAccessorW const &gamma_grad,
    GenericTensorAccessorW const &beta_grad);

void batch_norm_cleanup_kernel(
    DeviceType device_type,
    Allocator &allocator,
    std::optional<BatchNormPerDeviceState> &per_device_state);

} // namespace FlexFlow

#endif

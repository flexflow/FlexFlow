#ifndef _FLEXFLOW_OPS_KERNELS_POOL_2D_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_POOL_2D_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/pool_2d_per_device_state.dtg.h"
#include "op-attrs/ops/pool_2d_attrs.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "pcg/device_type.dtg.h"

namespace FlexFlow {

std::optional<Pool2DPerDeviceState>
    pool_2d_init_kernel(DeviceType device_type,
                        Pool2DAttrs const &attrs,
                        TensorShape const &input_shape,
                        TensorShape const &output_shape);

void pool_2d_forward_kernel(
    device_stream_t const &stream,
    device_handle_t const &handle,
    std::optional<Pool2DPerDeviceState> const &per_device_state,
    Pool2DAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &output);

void pool_2d_backward_kernel(
    device_stream_t const &stream,
    device_handle_t const &handle,
    std::optional<Pool2DPerDeviceState> const &per_device_state,
    Pool2DAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad);

void pool_2d_cleanup_kernel(
    DeviceType device_type,
    std::optional<Pool2DPerDeviceState> &per_device_state);

} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_POOL_2D_KERNELS_H

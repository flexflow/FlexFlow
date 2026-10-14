#ifndef _FLEXFLOW_OPS_KERNELS_POOL_2D_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_POOL_2D_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/ff_handle.h"
#include "kernels/pool_2d_per_device_state.dtg.h"
#include "op-attrs/activation.dtg.h"
#include "op-attrs/ops/pool_2d.h"
#include "pcg/device_type.dtg.h"

namespace FlexFlow {

std::optional<Pool2DPerDeviceState>
    pool2d_init_kernel(DeviceType device_type,
                       device_handle_t const &handle,
                       Pool2DAttrs const &attrs,
                       TensorShape const &input_shape);

void pool2d_forward_kernel(
    device_stream_t const &stream,
    std::optional<Pool2DPerDeviceState> const &per_device_state,
    Pool2DAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &output);

void pool2d_backward_kernel(
    device_stream_t const &stream,
    std::optional<Pool2DPerDeviceState> const &per_device_state,
    Pool2DAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad);

void cleanup_kernel(DeviceType device_type,
                    std::optional<Pool2DPerDeviceState> &per_device_state);

} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_POOL_2D_KERNELS_H

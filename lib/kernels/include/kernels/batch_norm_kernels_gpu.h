#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_BATCH_NORM_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_BATCH_NORM_KERNELS_GPU_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "kernels/batch_norm_per_device_state.dtg.h"
#include "kernels/device.h"
#include "op-attrs/ops/batch_norm_attrs.dtg.h"

namespace FlexFlow {

BatchNormPerDeviceState
    batch_norm_gpu_init_kernel(Allocator &allocator,
                               BatchNormAttrs const &attrs,
                               TensorShape const &input_shape,
                               TensorShape const &output_shape);

void batch_norm_gpu_forward_kernel(
    ffStream_t stream,
    PerDeviceFFHandle const &handle,
    BatchNormPerDeviceState const &per_device_state,
    BatchNormAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorR const &gamma,
    GenericTensorAccessorR const &beta,
    GenericTensorAccessorW const &output);

void batch_norm_gpu_backward_kernel(
    ffStream_t stream,
    PerDeviceFFHandle const &handle,
    BatchNormPerDeviceState const &per_device_state,
    BatchNormAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &gamma,
    GenericTensorAccessorW const &gamma_grad,
    GenericTensorAccessorW const &beta_grad);

void batch_norm_gpu_cleanup_kernel(Allocator &allocator,
                                   BatchNormPerDeviceState &per_device_state);

} // namespace FlexFlow

#endif

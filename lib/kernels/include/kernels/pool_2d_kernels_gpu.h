#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_POOL_2D_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_POOL_2D_KERNELS_GPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "kernels/pool_2d_per_device_state.dtg.h"
#include "op-attrs/ops/pool_2d_attrs.dtg.h"

namespace FlexFlow {

Pool2DPerDeviceState pool_2d_gpu_init_kernel(Pool2DAttrs const &attrs,
                                             TensorShape const &input_shape,
                                             TensorShape const &output_shape);

void pool_2d_gpu_forward_kernel(ffStream_t stream,
                                PerDeviceFFHandle const &handle,
                                Pool2DPerDeviceState const &per_device_state,
                                Pool2DAttrs const &attrs,
                                GenericTensorAccessorR const &input,
                                GenericTensorAccessorW const &output);

void pool_2d_gpu_backward_kernel(ffStream_t stream,
                                 PerDeviceFFHandle const &handle,
                                 Pool2DPerDeviceState const &per_device_state,
                                 Pool2DAttrs const &attrs,
                                 GenericTensorAccessorR const &output,
                                 GenericTensorAccessorR const &output_grad,
                                 GenericTensorAccessorR const &input,
                                 GenericTensorAccessorW const &input_grad);

void pool_2d_gpu_cleanup_kernel(Pool2DPerDeviceState &per_device_state);

} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_POOL_2D_KERNELS_GPU_H

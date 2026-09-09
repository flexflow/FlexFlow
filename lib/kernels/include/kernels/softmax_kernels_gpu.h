#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_SOFTMAX_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_SOFTMAX_KERNELS_GPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "kernels/softmax_per_device_state.dtg.h"
#include "op-attrs/ops/softmax_attrs.dtg.h"

namespace FlexFlow {

SoftmaxPerDeviceState softmax_gpu_init_kernel(SoftmaxAttrs const &attrs,
                                              TensorShape const &input_shape,
                                              TensorShape const &output_shape);

void softmax_gpu_forward_kernel(ffStream_t stream,
                                PerDeviceFFHandle const &handle,
                                SoftmaxPerDeviceState const &per_device_state,
                                SoftmaxAttrs const &attrs,
                                GenericTensorAccessorR const &input,
                                GenericTensorAccessorW const &output);

void softmax_gpu_backward_kernel(ffStream_t stream,
                                 PerDeviceFFHandle const &handle,
                                 SoftmaxPerDeviceState const &per_device_state,
                                 SoftmaxAttrs const &attrs,
                                 GenericTensorAccessorR const &output,
                                 GenericTensorAccessorR const &output_grad,
                                 GenericTensorAccessorR const &input,
                                 GenericTensorAccessorW const &input_grad);

void softmax_gpu_cleanup_kernel(SoftmaxPerDeviceState &per_device_state);

} // namespace FlexFlow

#endif

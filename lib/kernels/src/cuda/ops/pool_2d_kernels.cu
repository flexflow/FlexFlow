/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "internal/device.h"
#include "kernels/pool_2d_kernels_gpu.h"
#include "op-attrs/tensor_dims.h"
#include "utils/exception.h"

namespace FlexFlow {

static cudnnPoolingMode_t cudnn_pooling_mode_from_pool_op(PoolOp pool_type) {
  switch (pool_type) {
    case PoolOp::MAX:
      return CUDNN_POOLING_MAX;
    case PoolOp::AVG:
      return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    default:
      NOT_IMPLEMENTED();
  }
}

Pool2DPerDeviceState pool_2d_gpu_init_kernel(Pool2DAttrs const &attrs,
                                             TensorShape const &input_shape,
                                             TensorShape const &output_shape) {
  // Applying an activation as part of Pool2D is not currently implemented (it
  // used to be silently ignored). If you need it, please create an issue.
  ASSERT(!attrs.activation.has_value(),
         "Pool2D does not currently support fused activations",
         attrs.activation);

  ASSERT(get_num_dims(input_shape.dims) == num_tensor_dims_t{4_n},
         "Pool2D expects 4-dimensional (i.e., NCHW) input tensors",
         input_shape);
  ASSERT(get_num_dims(output_shape.dims) == num_tensor_dims_t{4_n},
         "Pool2D expects 4-dimensional (i.e., NCHW) output tensors",
         output_shape);

  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t outputTensor;
  ffPoolingDescriptor_t poolDesc;

  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));

  checkCUDNN(cudnnSetTensorDescriptorFromTensorShape(inputTensor, input_shape));

  checkCUDNN(cudnnSetPooling2dDescriptor(
      poolDesc,
      cudnn_pooling_mode_from_pool_op(attrs.pool_type),
      CUDNN_PROPAGATE_NAN,
      attrs.kernel_h.int_from_positive_int(),
      attrs.kernel_w.int_from_positive_int(),
      attrs.padding_h.unwrap_nonnegative(),
      attrs.padding_w.unwrap_nonnegative(),
      attrs.stride_h.int_from_positive_int(),
      attrs.stride_w.int_from_positive_int()));

  int n, c, h, w;
  checkCUDNN(
      cudnnGetPooling2dForwardOutputDim(poolDesc, inputTensor, &n, &c, &h, &w));

  ASSERT(dim_at_idx(output_shape.dims, ff_dim_t{0_n}) == positive_int{n});
  ASSERT(dim_at_idx(output_shape.dims, ff_dim_t{1_n}) == positive_int{c});
  ASSERT(dim_at_idx(output_shape.dims, ff_dim_t{2_n}) == positive_int{h});
  ASSERT(dim_at_idx(output_shape.dims, ff_dim_t{3_n}) == positive_int{w});

  checkCUDNN(
      cudnnSetTensorDescriptorFromTensorShape(outputTensor, output_shape));

  return Pool2DPerDeviceState{
      /*inputTensor=*/inputTensor,
      /*outputTensor=*/outputTensor,
      /*poolDesc=*/poolDesc,
  };
}

void pool_2d_gpu_forward_kernel(cudaStream_t stream,
                                PerDeviceFFHandle const &handle,
                                Pool2DPerDeviceState const &per_device_state,
                                Pool2DAttrs const &attrs,
                                GenericTensorAccessorR const &input,
                                GenericTensorAccessorW const &output) {
  checkCUDNN(cudnnSetStream(handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnPoolingForward(handle.dnn,
                                 per_device_state.poolDesc,
                                 &alpha,
                                 per_device_state.inputTensor,
                                 input.ptr,
                                 &beta,
                                 per_device_state.outputTensor,
                                 output.ptr));
}

void pool_2d_gpu_backward_kernel(cudaStream_t stream,
                                 PerDeviceFFHandle const &handle,
                                 Pool2DPerDeviceState const &per_device_state,
                                 Pool2DAttrs const &attrs,
                                 GenericTensorAccessorR const &output,
                                 GenericTensorAccessorR const &output_grad,
                                 GenericTensorAccessorR const &input,
                                 GenericTensorAccessorW const &input_grad) {
  checkCUDNN(cudnnSetStream(handle.dnn, stream));

  // NOTE: beta is 1.0 so that input_grad is accumulated into
  float alpha = 1.0f, beta = 1.0f;
  checkCUDNN(cudnnPoolingBackward(handle.dnn,
                                  per_device_state.poolDesc,
                                  &alpha,
                                  per_device_state.outputTensor,
                                  output.ptr,
                                  per_device_state.outputTensor,
                                  output_grad.ptr,
                                  per_device_state.inputTensor,
                                  input.ptr,
                                  &beta,
                                  per_device_state.inputTensor,
                                  input_grad.ptr));
}

void pool_2d_gpu_cleanup_kernel(Pool2DPerDeviceState &per_device_state) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow

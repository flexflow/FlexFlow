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
#include "kernels/softmax_kernels_gpu.h"
#include "op-attrs/ff_dim_t.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "utils/exception.h"

namespace FlexFlow {

SoftmaxPerDeviceState softmax_gpu_init_kernel(SoftmaxAttrs const &attrs,
                                              TensorShape const &input_shape,
                                              TensorShape const &output_shape) {
  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t outputTensor;
  ffTensorDescriptor_t outputGradTensor;

  TensorShape shape = require_same(input_shape, output_shape);

  positive_int num_outer_elements =
      get_num_elements(slice_tensor_dims(shape.dims, ff_dim_t{0_n}, attrs.dim));
  positive_int softmax_dim_size = dim_at_idx(shape.dims, attrs.dim);
  positive_int num_inner_elements = get_num_elements(
      slice_tensor_dims(shape.dims, add_to_ff_dim(attrs.dim, 1), std::nullopt));

  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnSetTensor4dDescriptor(
      inputTensor,
      CUDNN_TENSOR_NCHW,
      ff_to_cudnn_datatype(shape.data_type),
      /*n=*/num_outer_elements.int_from_positive_int(),
      /*c=*/softmax_dim_size.int_from_positive_int(),
      /*h=*/num_inner_elements.int_from_positive_int(),
      /*w=*/1));

  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnSetTensor4dDescriptor(
      outputTensor,
      CUDNN_TENSOR_NCHW,
      ff_to_cudnn_datatype(shape.data_type),
      /*n=*/num_outer_elements.int_from_positive_int(),
      /*c=*/softmax_dim_size.int_from_positive_int(),
      /*h=*/num_inner_elements.int_from_positive_int(),
      /*w=*/1));

  checkCUDNN(cudnnCreateTensorDescriptor(&outputGradTensor));
  checkCUDNN(cudnnSetTensor4dDescriptor(
      outputGradTensor,
      CUDNN_TENSOR_NCHW,
      ff_to_cudnn_datatype(shape.data_type),
      /*n=*/num_outer_elements.int_from_positive_int(),
      /*c=*/softmax_dim_size.int_from_positive_int(),
      /*h=*/num_inner_elements.int_from_positive_int(),
      /*w=*/1));

  return SoftmaxPerDeviceState{
      /*inputTensor=*/inputTensor,
      /*outputTensor=*/outputTensor,
      /*outputGradTensor=*/outputGradTensor,
  };
}

void softmax_gpu_forward_kernel(ffStream_t stream,
                                PerDeviceFFHandle const &handle,
                                SoftmaxPerDeviceState const &per_device_state,
                                SoftmaxAttrs const &attrs,
                                GenericTensorAccessorR const &input,
                                GenericTensorAccessorW const &output) {
  checkCUDNN(cudnnSetStream(handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnSoftmaxForward(handle.dnn,
                                 CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_CHANNEL,
                                 &alpha,
                                 per_device_state.inputTensor,
                                 input.get_float_ptr(),
                                 &beta,
                                 per_device_state.inputTensor,
                                 output.get_float_ptr()));
}

void softmax_gpu_backward_kernel(ffStream_t stream,
                                 PerDeviceFFHandle const &handle,
                                 SoftmaxPerDeviceState const &per_device_state,
                                 SoftmaxAttrs const &attrs,
                                 GenericTensorAccessorR const &output,
                                 GenericTensorAccessorR const &output_grad,
                                 GenericTensorAccessorR const &input,
                                 GenericTensorAccessorW const &input_grad) {
  checkCUDNN(cudnnSetStream(handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnSoftmaxBackward(handle.dnn,
                                  CUDNN_SOFTMAX_ACCURATE,
                                  CUDNN_SOFTMAX_MODE_CHANNEL,
                                  &alpha,
                                  per_device_state.inputTensor,
                                  output.get_float_ptr(),
                                  per_device_state.inputTensor,
                                  output_grad.get_float_ptr(),
                                  &beta,
                                  per_device_state.inputTensor,
                                  input_grad.get_float_ptr()));
}

void softmax_gpu_cleanup_kernel(SoftmaxPerDeviceState &per_device_state) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow

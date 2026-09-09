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
#include "kernels/batch_norm_kernels_gpu.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/require_same.h"
#include <vector>

namespace FlexFlow {

static positive_int get_num_channels(TensorShape const &shape) {
  return dim_at_idx(shape.dims, ff_dim_t{1_n});
}

BatchNormPerDeviceState
    batch_norm_gpu_init_kernel(Allocator &allocator,
                               BatchNormAttrs const &attrs,
                               TensorShape const &input_shape,
                               TensorShape const &output_shape) {
  // Applying an activation as part of BatchNorm is not currently implemented
  // (the forward kernel used to silently ignore it while the backward kernel
  // applied it). If you need it, please create an issue.
  ASSERT(!attrs.relu,
         "BatchNorm does not currently support a fused relu activation");

  ASSERT(attrs.affine,
         "BatchNorm currently only supports attrs.affine = true. "
         "If you need this feature, please create an issue.");

  TensorShape shape = require_same(input_shape, output_shape);

  ASSERT(get_num_dims(shape.dims) == num_tensor_dims_t{4_n},
         "BatchNorm currently only supports 4-dimensional (i.e., NCHW) "
         "tensors. If you need this feature, please create an issue.",
         shape);

  ASSERT(attrs.eps >= CUDNN_BN_MIN_EPSILON,
         "cuDNN requires BatchNorm eps to be at least CUDNN_BN_MIN_EPSILON",
         attrs.eps,
         CUDNN_BN_MIN_EPSILON);

  int num_channels = get_num_channels(shape).int_from_positive_int();

  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t outputTensor;
  ffTensorDescriptor_t biasTensor;

  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));

  ffBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION >= 7000
  mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#endif

  checkCUDNN(cudnnSetTensorDescriptorFromTensorShape(inputTensor, input_shape));
  checkCUDNN(
      cudnnSetTensorDescriptorFromTensorShape(outputTensor, output_shape));
  checkCUDNN(cudnnSetTensor4dDescriptor(biasTensor,
                                        CUDNN_TENSOR_NCHW,
                                        ff_to_cudnn_datatype(shape.data_type),
                                        /*n=*/1,
                                        /*c=*/num_channels,
                                        /*h=*/1,
                                        /*w=*/1));

  // Allocate memory for runningMean, runningVar, saveMean and saveVar as a
  // single contiguous block (deallocated by batch_norm_gpu_cleanup_kernel).
  float *runningMean = static_cast<float *>(
      allocator.allocate(sizeof(float) * num_channels * 4));
  float *runningVar = runningMean + num_channels;
  float *saveMean = runningVar + num_channels;
  float *saveVar = saveMean + num_channels;

  // Match the PyTorch initialization of running_mean = 0 and running_var = 1.
  std::vector<float> initial_running_stats(num_channels * 2);
  std::fill(initial_running_stats.begin(),
            initial_running_stats.begin() + num_channels,
            0.0f);
  std::fill(initial_running_stats.begin() + num_channels,
            initial_running_stats.end(),
            1.0f);
  checkCUDA(cudaMemcpy(runningMean,
                       initial_running_stats.data(),
                       sizeof(float) * num_channels * 2,
                       cudaMemcpyHostToDevice));

  return BatchNormPerDeviceState{
      /*inputTensor=*/inputTensor,
      /*outputTensor=*/outputTensor,
      /*biasTensor=*/biasTensor,
      /*mode=*/mode,
      /*runningMean=*/runningMean,
      /*runningVar=*/runningVar,
      /*saveMean=*/saveMean,
      /*saveVar=*/saveVar,
  };
}

void batch_norm_gpu_forward_kernel(
    cudaStream_t stream,
    PerDeviceFFHandle const &handle,
    BatchNormPerDeviceState const &per_device_state,
    BatchNormAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorR const &gamma,
    GenericTensorAccessorR const &beta,
    GenericTensorAccessorW const &output) {
  checkCUDNN(cudnnSetStream(handle.dnn, stream));

  // NOTE: attrs.momentum = std::nullopt means "use a cumulative moving
  // average", which cuDNN cannot express, so we fall back to fully replacing
  // the running statistics on each call. The running statistics are currently
  // never read back, so this only matters once inference mode is supported.
  double exponential_average_factor = attrs.momentum.value_or(1.0);

  float alpha = 1.0f, beta_coeff = 0.0f;
  checkCUDNN(
      cudnnBatchNormalizationForwardTraining(handle.dnn,
                                             per_device_state.mode,
                                             &alpha,
                                             &beta_coeff,
                                             per_device_state.inputTensor,
                                             input.ptr,
                                             per_device_state.outputTensor,
                                             output.ptr,
                                             per_device_state.biasTensor,
                                             gamma.ptr,
                                             beta.ptr,
                                             exponential_average_factor,
                                             per_device_state.runningMean,
                                             per_device_state.runningVar,
                                             attrs.eps,
                                             per_device_state.saveMean,
                                             per_device_state.saveVar));
}

void batch_norm_gpu_backward_kernel(
    cudaStream_t stream,
    PerDeviceFFHandle const &handle,
    BatchNormPerDeviceState const &per_device_state,
    BatchNormAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &gamma,
    GenericTensorAccessorW const &gamma_grad,
    GenericTensorAccessorW const &beta_grad) {
  checkCUDNN(cudnnSetStream(handle.dnn, stream));

  // NOTE: the beta coefficients are 1.0 so that the gradients are accumulated
  // into rather than overwritten
  float alpha_data = 1.0f, beta_data = 1.0f;
  float alpha_param = 1.0f, beta_param = 1.0f;
  checkCUDNN(cudnnBatchNormalizationBackward(handle.dnn,
                                             per_device_state.mode,
                                             &alpha_data,
                                             &beta_data,
                                             &alpha_param,
                                             &beta_param,
                                             per_device_state.inputTensor,
                                             input.ptr,
                                             per_device_state.outputTensor,
                                             output_grad.ptr,
                                             per_device_state.inputTensor,
                                             input_grad.ptr,
                                             per_device_state.biasTensor,
                                             gamma.ptr,
                                             gamma_grad.ptr,
                                             beta_grad.ptr,
                                             attrs.eps,
                                             per_device_state.saveMean,
                                             per_device_state.saveVar));
}

void batch_norm_gpu_cleanup_kernel(Allocator &allocator,
                                   BatchNormPerDeviceState &per_device_state) {
  allocator.deallocate(per_device_state.runningMean);
  checkCUDNN(cudnnDestroyTensorDescriptor(per_device_state.inputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(per_device_state.outputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(per_device_state.biasTensor));
}

} // namespace FlexFlow

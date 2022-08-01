/* Copyright 2020 Stanford
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

#include "flexflow/model.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

using namespace Legion;

__global__ void
sparse_categorical_crossentropy_loss_backward(float *logit_grad,
                                              const int *label,
                                              coord_t num_samples,
                                              coord_t num_classes,
                                              const int k) {
  CUDA_KERNEL_LOOP(i, num_samples) {
    int label_idx = label[i / k];
    logit_grad[i * num_classes + label_idx] -= 1.0f;
  }
}

__global__ void categorical_crossentropy_loss_backward(float *logit_grad,
                                                       const float *logit,
                                                       const float *label,
                                                       coord_t num_elements) {
  CUDA_KERNEL_LOOP(i, num_elements) { logit_grad[i] = logit[i] - label[i]; }
}

__global__ void mean_squared_error_avg_loss_backward(float *logit_grad,
                                                     const float *logit,
                                                     const float *label,
                                                     coord_t num_elements) {
  CUDA_KERNEL_LOOP(i, num_elements) { logit_grad[i] = logit[i] - label[i]; }
}

void Loss::sparse_categorical_crossentropy_loss_backward_kernel_wrapper(
    float *logit_grad_ptr,
    const float *logit_ptr,
    const int *label_ptr,
    size_t logit_volume,
    size_t logit_grad_volume,
    int num_samples,
    int num_classes,
    int k,
    float scale_factor) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDA(cudaMemcpy(logit_grad_ptr,
                       logit_ptr,
                       logit_volume * sizeof(float),
                       cudaMemcpyDeviceToDevice));
  sparse_categorical_crossentropy_loss_backward<<<GET_BLOCKS(num_samples),
                                                  CUDA_NUM_THREADS,
                                                  0,
                                                  stream>>>(
      logit_grad_ptr, label_ptr, num_samples, num_classes, k);
  // Scale logit gradients by op->scale_factor
  scale_kernel<<<GET_BLOCKS(logit_grad_volume), CUDA_NUM_THREADS, 0, stream>>>(
      logit_grad_ptr, logit_grad_volume, 0, scale_factor * k);
}

void Loss::categorical_crossentropy_loss_backward_kernel_wrapper(
    float *logit_grad_ptr,
    const float *logit_ptr,
    const float *label_ptr,
    size_t logit_volume,
    size_t logit_grad_volume,
    float scale_factor) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  categorical_crossentropy_loss_backward<<<GET_BLOCKS(logit_volume),
                                           CUDA_NUM_THREADS,
                                           0,
                                           stream>>>(
      logit_grad_ptr, logit_ptr, label_ptr, logit_volume);
  // Scale logit gradients by loss->scale_factor
  scale_kernel<<<GET_BLOCKS(logit_grad_volume), CUDA_NUM_THREADS, 0, stream>>>(
      logit_grad_ptr, logit_grad_volume, 0, scale_factor);
}

void Loss::mean_squared_error_avg_loss_backward_kernel_wrapper(
    float *logit_grad_ptr,
    const float *logit_ptr,
    const float *label_ptr,
    size_t logit_volume,
    size_t logit_grad_volume,
    float scale_factor) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  mean_squared_error_avg_loss_backward<<<GET_BLOCKS(logit_volume),
                                         CUDA_NUM_THREADS,
                                         0,
                                         stream>>>(
      logit_grad_ptr, logit_ptr, label_ptr, logit_volume);
  // Scale logit gradients by loss->scale_factor
  scale_kernel<<<GET_BLOCKS(logit_grad_volume), CUDA_NUM_THREADS, 0, stream>>>(
      logit_grad_ptr, logit_grad_volume, 0, scale_factor);
}

}; // namespace FlexFlow

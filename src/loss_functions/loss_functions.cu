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

#include "flexflow/model.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

int const MASK_TOKEN = -100;
using namespace Legion;

__global__ void
    sparse_categorical_crossentropy_loss_backward(float *logit_grad,
                                                  int const *label,
                                                  coord_t num_samples,
                                                  coord_t num_classes,
                                                  int const k) {
  CUDA_KERNEL_LOOP(i, num_samples) {
    int label_idx = label[i / k];
    logit_grad[i * num_classes + label_idx] -= 1.0f;
  }
}

__global__ void
    sparse_categorical_crossentropy_loss_backward_with_mask(float *logit_grad,
                                                            int const *label,
                                                            coord_t num_samples,
                                                            coord_t num_classes,
                                                            int const k,
                                                            float *num) {
  CUDA_KERNEL_LOOP(i, num_samples * num_classes) {
    int sample_id = i / num_classes;
    int label_idx = label[i / (k * num_classes)];
    if (label_idx != MASK_TOKEN && (i == sample_id * num_classes + label_idx)) {
      logit_grad[i] -= 1.0f;
      atomicAdd(&num[0], 1.0f);
    } else if (label_idx == MASK_TOKEN) {
      logit_grad[i] = 0.0f;
    }
  }
}

__global__ void categorical_crossentropy_loss_backward(float *logit_grad,
                                                       float const *logit,
                                                       float const *label,
                                                       coord_t num_elements) {
  CUDA_KERNEL_LOOP(i, num_elements) {
    logit_grad[i] = logit[i] - label[i];
  }
}

__global__ void mean_squared_error_avg_loss_backward(float *logit_grad,
                                                     float const *logit,
                                                     float const *label,
                                                     coord_t num_elements) {
  CUDA_KERNEL_LOOP(i, num_elements) {
    logit_grad[i] = logit[i] - label[i];
  }
}

__global__ void identity_loss_backward(float *loss_grad,
                                       float const *loss,
                                       coord_t num_elements) {
  CUDA_KERNEL_LOOP(i, num_elements) {
    loss_grad[i] = 1.0f;
  }
}

void Loss::sparse_categorical_crossentropy_loss_backward_kernel_wrapper(
    float *logit_grad_ptr,
    float const *logit_ptr,
    int const *label_ptr,
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
  // calculate the scale factor inside kernel;
  assert(scale_factor == 1.0f);
  float *num;
  checkCUDA(cudaMalloc(&num, sizeof(float)));
  float effective_tokens;
  int parallelism = num_samples * num_classes;
  // sparse_categorical_crossentropy_loss_backward<<<GET_BLOCKS(num_samples),
  //                                                 CUDA_NUM_THREADS,
  //                                                 0,
  //                                                 stream>>>(
  //     logit_grad_ptr, label_ptr, num_samples, num_classes, k, num);
  sparse_categorical_crossentropy_loss_backward_with_mask<<<
      GET_BLOCKS(parallelism),
      CUDA_NUM_THREADS,
      0,
      stream>>>(logit_grad_ptr, label_ptr, num_samples, num_classes, k, num);
  cudaMemcpy(&effective_tokens, num, sizeof(float), cudaMemcpyDeviceToHost);
  scale_kernel<<<GET_BLOCKS(logit_grad_volume), CUDA_NUM_THREADS, 0, stream>>>(
      logit_grad_ptr, logit_grad_volume, 0, 1.0f / effective_tokens);
}

void Loss::categorical_crossentropy_loss_backward_kernel_wrapper(
    float *logit_grad_ptr,
    float const *logit_ptr,
    float const *label_ptr,
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
    float const *logit_ptr,
    float const *label_ptr,
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

void Loss::identity_loss_backward_kernel_wrapper(float *loss_grad_ptr,
                                                 float const *loss_ptr,
                                                 size_t loss_volume,
                                                 size_t loss_grad_volume,
                                                 float scale_factor) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  identity_loss_backward<<<GET_BLOCKS(loss_volume),
                           CUDA_NUM_THREADS,
                           0,
                           stream>>>(loss_grad_ptr, loss_ptr, loss_volume);
  // Scale logit gradients by loss->scale_factor
  scale_kernel<<<GET_BLOCKS(loss_grad_volume), CUDA_NUM_THREADS, 0, stream>>>(
      loss_grad_ptr, loss_grad_volume, 0, scale_factor);
}

}; // namespace FlexFlow

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

#include "flexflow/ops/groupby.h"
#include "flexflow/utils/cuda_helper.h"
#include <math.h>
#include <stdio.h>

#define MAX_K 4
#define MAX_BATCH_SIZE 64
#define MAX_N 12

namespace FlexFlow {

__global__ void
    gb_forward_kernel(float const *input,
                      int const *exp_assign,
                      float **outputs,
                      int n,       // num experts
                      int k,       // chosen experts
                      float alpha, // factor additional memory assigned
                      int batch_size,
                      int data_dim) {
  __shared__ float *chosen_exp_preds[MAX_K * MAX_BATCH_SIZE];

  // Get pred pointers, single thread per block
  if (threadIdx.x == 0) {
    int exp_tensor_rows = ceil(alpha * k / n * batch_size);
    int expert_idx[MAX_N] = {0};
    for (int i = 0; i < k * batch_size; i++) {
      // Get pointer to chosen expert predictions
      int expert = exp_assign[i];
      if (expert_idx[expert] >= exp_tensor_rows) {
        // dropped sample
        chosen_exp_preds[i] = 0;
        continue;
      }
      chosen_exp_preds[i] = outputs[expert] + expert_idx[expert] * data_dim;
      expert_idx[expert]++;
    }
  }

  __syncthreads();

  // compute output
  CUDA_KERNEL_LOOP(i, k * batch_size * data_dim) {
    if (chosen_exp_preds[i / data_dim] != 0) {
      float a = input[(i / (k * data_dim)) * data_dim + i % data_dim];
      chosen_exp_preds[i / data_dim][i % data_dim] = a;
    }
  }
}

__global__ void
    gb_backward_kernel(float *input_grad,
                       int const *exp_assign,
                       float **output_grads,
                       int n,       // num experts
                       int k,       // chosen experts
                       float alpha, // factor additional memory assigned
                       int batch_size,
                       int data_dim) {
  __shared__ float *chosen_exp_grads[MAX_K * MAX_BATCH_SIZE];
  assert(k <= MAX_K);
  assert(batch_size <= MAX_BATCH_SIZE);
  assert(n <= MAX_N);

  // Get pred pointers, single thread
  if (threadIdx.x == 0) {
    int exp_tensor_rows = ceil(alpha * k / n * batch_size);
    int expert_idx[MAX_N] = {0};
    for (int i = 0; i < k * batch_size; i++) {
      // Get pointer to chosen expert predictions
      int expert = exp_assign[i];
      if (expert_idx[expert] >= exp_tensor_rows) {
        // dropped sample
        chosen_exp_grads[i] = nullptr;
        continue;
      }
      chosen_exp_grads[i] =
          output_grads[expert] + expert_idx[expert] * data_dim;
      expert_idx[expert]++;
    }
  }

  __syncthreads();
  // compute output
  CUDA_KERNEL_LOOP(i, k * batch_size * data_dim) {
    if (chosen_exp_grads[i / data_dim] != nullptr) {
      input_grad[(i / (k * data_dim)) * data_dim + i % data_dim] =
          chosen_exp_grads[i / data_dim][i % data_dim];
    }
  }
}

/*static*/
void Group_by::forward_kernel_wrapper(GroupByMeta const *m,
                                      float const *input,
                                      int const *exp_assign,
                                      float **outputs,
                                      int n, // num experts
                                      int k, // chosen experts
                                      int batch_size,
                                      int data_dim) {
  // TODO: why cublas/cudnn stream is needed here?

  float alpha = m->alpha;

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  // call forward kernel
  cudaMemcpyAsync(m->dev_region_ptrs,
                  outputs,
                  n * sizeof(float *),
                  cudaMemcpyHostToDevice,
                  stream);

  gb_forward_kernel<<<GET_BLOCKS(batch_size * k * data_dim),
                      min(CUDA_NUM_THREADS, (int)(batch_size * k * data_dim)),
                      0,
                      stream>>>(
      input, exp_assign, m->dev_region_ptrs, n, k, alpha, batch_size, data_dim);
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[GroupBy] forward time = %.2lfms\n", elapsed);
  }
}

void Group_by::backward_kernel_wrapper(GroupByMeta const *m,
                                       float *input_grad,
                                       int const *exp_assign,
                                       float **output_grads,
                                       int n, // num experts
                                       int k, // chosen experts
                                       int batch_size,
                                       int data_dim) {

  float alpha = m->alpha;

  // TODO: why cublas/cudnn stream is needed here
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  // call forward kernel
  cudaMemcpyAsync(m->dev_region_ptrs,
                  output_grads,
                  n * sizeof(float *),
                  cudaMemcpyHostToDevice,
                  stream);
  gb_backward_kernel<<<GET_BLOCKS(batch_size * k * data_dim),
                       min(CUDA_NUM_THREADS, (int)(batch_size * k * data_dim)),
                       0,
                       stream>>>(input_grad,
                                 exp_assign,
                                 m->dev_region_ptrs,
                                 n,
                                 k,
                                 alpha,
                                 batch_size,
                                 data_dim);
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[GroupBy] backward time = %.2lfms\n", elapsed);
  }
}

GroupByMeta::GroupByMeta(FFHandler handler, int n, float _alpha)
    : OpMeta(handler), alpha(_alpha) {
  checkCUDA(cudaMalloc(&dev_region_ptrs, n * sizeof(float *)));
}
GroupByMeta::~GroupByMeta(void) {
  checkCUDA(cudaFree(&dev_region_ptrs));
}

}; // namespace FlexFlow

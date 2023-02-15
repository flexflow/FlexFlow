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

#include "flexflow/ops/experts.h"
#include "flexflow/utils/cuda_helper.h"
#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT 1
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

namespace FlexFlow {

__global__ void experts_forward_kernel1(int data_dim,
                                        int num_chosen_experts,
                                        int num_tokens,
                                        int num_experts_per_block,
                                        int experts_start_idx,
                                        int expert_capacity,
                                        float *tokens_array,
                                        float const *input,
                                        int const *indices,
                                        int *replicated_indices) {

  // initialize tokens_array with replicated tokens
  CUDA_KERNEL_LOOP(i, data_dim * num_tokens * num_chosen_experts) {
    int token_index = i / (data_dim * num_chosen_experts);
    int chosen_exp_index = (i % (data_dim * num_chosen_experts)) / data_dim;
    int data_dim_index = (i % (data_dim * num_chosen_experts)) % data_dim;
    assert(i == data_dim * num_chosen_experts * token_index +
                    data_dim * chosen_exp_index + data_dim_index);
    int j = data_dim * token_index + data_dim_index;
    tokens_array[i] = input[j];
    int k = token_index * num_chosen_experts + chosen_exp_index;
    replicated_indices[i] = indices[k];
  }
}

/*static*/
void Experts::forward_kernel_wrapper(ExpertsMeta const *m,
                                     float const *input,
                                     int const *indices,
                                     float const *topk_gate_preds,
                                     float *output,
                                     float const **weights,
                                     int chosen_experts,
                                     int batch_size,
                                     int out_dim) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  int num_experts_per_block = m->num_experts;
  int experts_start_idx = m->experts_start_idx;
  // bool use_bias = m->use_bias;
  // ActiMode activation = m->activation;
  int data_dim = m->data_dim;
  int num_chosen_experts = m->num_chosen_experts;
  int num_tokens = m->effective_batch_size;
  int expert_capacity = m->expert_capacity;

  assert(chosen_experts == num_chosen_experts);
  assert(num_tokens == batch_size);

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  int kernel1_parallelism = data_dim * num_tokens * num_chosen_experts;
  experts_forward_kernel1<<<GET_BLOCKS(kernel1_parallelism),
                            min(CUDA_NUM_THREADS, (int)kernel1_parallelism),
                            0,
                            stream>>>(data_dim,
                                      num_chosen_experts,
                                      num_tokens,
                                      num_experts_per_block,
                                      experts_start_idx,
                                      expert_capacity,
                                      m->dev_sorted_tokens,
                                      input,
                                      indices,
                                      m->dev_replicated_indices);

  // sort the tokens by expert
  thrust::device_ptr<float> thrust_tokens_ptr =
      thrust::device_pointer_cast(m->dev_sorted_tokens);
  thrust::device_ptr<int> thrust_indices_ptr =
      thrust::device_pointer_cast(m->dev_replicated_indices);
  thrust::stable_sort_by_key(thrust::device,
                             thrust_indices_ptr,
                             thrust_indices_ptr +
                                 num_chosen_experts * num_tokens * data_dim,
                             thrust_tokens_ptr,
                             thrust::greater<int>());

  // get index of each expert block (containing all tokens assigned to the same
  // expert)
  thrust::device_ptr<int> thrust_exp_slice_ptr =
      thrust::device_pointer_cast(m->dev_exp_slice_indices);
  thrust::device_ptr<int> thrust_exp_slice_ptr_end =
      thrust_exp_slice_ptr + num_chosen_experts * num_tokens * data_dim;
  thrust::sequence(thrust_exp_slice_ptr, thrust_exp_slice_ptr_end);
  int non_zero_tokens_experts =
      (thrust::unique_by_key(thrust_indices_ptr,
                             thrust_indices_ptr +
                                 num_chosen_experts * num_tokens * data_dim,
                             thrust_exp_slice_ptr))
          .first -
      thrust_indices_ptr;
  thrust::device_ptr<float> thrust_dev_tokens_in_use_ptr =
      thrust::device_pointer_cast(m->dev_tokens_in_use);

  thrust::copy_n(thrust_exp_slice_ptr,
                 non_zero_tokens_experts,
                 thrust_dev_tokens_in_use_ptr);

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[Experts] forward time = %.2lfms\n", elapsed);
  }
}

ExpertsMeta::ExpertsMeta(FFHandler handler,
                         int _num_experts,
                         int _experts_start_idx,
                         int _data_dim,
                         int _effective_batch_size,
                         int _num_chosen_experts,
                         float _alpha,
                         bool _use_bias,
                         ActiMode _activation)
    : OpMeta(handler), num_experts(_num_experts),
      experts_start_idx(_experts_start_idx), data_dim(_data_dim),
      effective_batch_size(_effective_batch_size),
      num_chosen_experts(_num_chosen_experts), alpha(_alpha),
      use_bias(_use_bias), activation(_activation) {
  expert_capacity =
      ceil(alpha * num_chosen_experts / num_experts * effective_batch_size);
  checkCUDA(cudaMalloc(&dev_sorted_tokens,
                       data_dim * effective_batch_size * num_chosen_experts *
                           sizeof(float)));
  checkCUDA(cudaMalloc(&dev_replicated_indices,
                       data_dim * effective_batch_size * num_chosen_experts *
                           sizeof(int)));
  checkCUDA(cudaMalloc(&dev_exp_slice_indices, num_experts * sizeof(int)));
  checkCUDA(
      cudaMalloc(&dev_tokens_in_use,
                 data_dim * expert_capacity * num_experts * sizeof(float)));
}
ExpertsMeta::~ExpertsMeta(void) {
  checkCUDA(cudaFree(&dev_sorted_tokens));
  checkCUDA(cudaFree(&dev_replicated_indices));
  checkCUDA(cudaFree(&dev_exp_slice_indices));
  checkCUDA(cudaFree(&dev_tokens_in_use));
}

}; // namespace FlexFlow

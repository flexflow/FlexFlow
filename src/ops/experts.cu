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

namespace FlexFlow {

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

  int expert_capacity =
      ceil(m->alpha * chosen_experts / m->num_experts * batch_size);

  int num_experts = m->num_experts;
  // int expert_start_index = experts_start_idx;
  bool use_bias = m->use_bias;
  // ActiMode activation = m->activation;

  size_t input_volume = sizeof(input) / sizeof(input[0]);
  size_t num_tokens = input_volume / data_dim;

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  // cudaMemcpy(m->dev_weights,
  //            weights,
  //            num_experts * (1 + use_bias) * sizeof(float *),
  //            cudaMemcpyHostToDevice);

  // ##########################################################################################################################

  /** TODO: launch one or more kernel(s) to do the following:
   * 1. sort the tokens by expert to which they are assigned. This will require
   * replicating tokens when chosen_experts > 1
   * 2. matrix multiply (you can use cublasGemmEx) each slice of tokens with the
   * corresponding expert's weights tensor. Add the bias.
   *      - you can obtain the slice by selecting the tokens between the index
   * where the expert i starts and min(i+expert_capacity, index where expert i+1
   * starts)
   * 3. reorder the outputs by token, and aggregate the outputs of multiple
   * experts for the same token by computing an average weighted by the
   * appropriate coefficient from the topk_gate_preds matrix.
   */

  // 1. port the 2d matrix into Thrust
  thrust::device_vector< float > input_tokens1(input, input + input_volume);
  thrust::device_vector< float > input_tokens2(input, input + input_volume);
  thrust::device_vector< float > replicated_tokens(k*input_volume);
  // 2. replicate the tokens (assuming k=2 here)
  thrust::device_vector< int > sorting_keys1(input_volume);
  thrust::sequence(sorting_keys1.begin(), sorting_keys1.end());
  // divide sorting_keys1 by data_dim and then multiply by k
  thrust::transform(sorting_keys1.begin(), sorting_keys1.end(), thrust::make_constant_iterator(data_dim), sorting_keys1.begin(), thrust::divide<int>());
  thrust::transform(sorting_keys1.begin(), sorting_keys1.end(), thrust::make_constant_iterator(k), sorting_keys1.begin(), thrust::multiply<int>());
  // obtain the i-th sorting keys by adding +1 to the i-1 sorting keys 
  thrust::device_vector< int > sorting_keys2(input_volume);
  thrust::copy(sorting_keys1.begin(), sorting_keys1.end(), sorting_keys2.begin());
  thrust::transform(sorting_keys1.begin(), sorting_keys1.end(), thrust::make_constant_iterator(1), sorting_keys1.begin(), thrust::plus<int>());
  // populate the replicated_tokens vector with k side-by-side copies of each token
  thrust::device_vector< int > merged_keys(2*input_volume);
  thrust::merge_by_key(sorting_keys1.begin(), sorting_keys1.end(), sorting_keys2.begin(), sorting_keys2.end(), input_tokens1, input_tokens2, merged_keys, replicated_tokens, thrust::less<int>())
  // 3. sort the tokens by expert index to which they are assigned
  thrust::device_vector< int > expert_assignments(indices, indices + k*num_tokens);
  thrust::sort_by_key(expert_assignments, expert_assignments + k*num_tokens, replicated_tokens);

  // 4. matrix multiply each slice of min(expert_capacity, end_of_expert_slice) tokens by the corresponding weight
  
  // get list of experts (in this block) receiving non-zero tokens
  thrust::device_vector< int > experts_in_use(k*num_tokens);
  int tot_exps = thrust::unique_copy(expert_assignments.begin(), expert_assignments.end(), experts_in_use) - experts_in_use.begin();
  struct is_expert_in_block {
    __host__ __device__
    bool operator()(const int x)
    {
      return x >= experts_start_idx && x < experts_start_idx+num_experts;
    }
  };
  int n_used_experts = thrust::remove_if(experts_in_use.begin(), experts_in_use.begin() + tot_exps, is_expert_in_block()) - experts_in_use.begin();

  // TODO: pad replicated tokens array:

  // get the indexes of each slice of tokens
  thrust::device_vector< int > slice_indices(k*num_tokens);
  thrust::sequence(slice_indices.begin(), slice_indices.end());
  int num_slices = (thrust::unique_by_key(expert_assignments.begin(), expert_assignments.end(), slice_indices.begin())).first - expert_assignments.begin();

  // ##########################################################################################################################

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
                         float _alpha,
                         bool _use_bias,
                         ActiMode _activation)
    : OpMeta(handler), num_experts(_num_experts),
      experts_start_idx(_experts_start_idx), alpha(_alpha), use_bias(_use_bias),
      activation(_activation) {
  //checkCUDA(
  //    cudaMalloc(&dev_weights, num_experts * (1 + use_bias) * sizeof(float *)));
}
ExpertsMeta::~ExpertsMeta(void) {
  //checkCUDA(cudaFree(&dev_weights));
}

}; // namespace FlexFlow

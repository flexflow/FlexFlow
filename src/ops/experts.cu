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
//#include <thrust/device_vector.h>

namespace FlexFlow {

// struct divide_functor {
//   __host__ __device__ float operator()(int const &x, int const &y) const {
//     return x / y;
//   }
// };

// struct multiply_functor {
//   __host__ __device__ float operator()(int const &x, int const &y) const {
//     return x * y;
//   }
// };

__global__ void experts_forward_kernel1(int data_dim,
                                        int num_chosen_experts,
                                        int num_tokens,
                                        float *tokens_array,
                                        float const *input) {
  // initialize tokens_array with replicated tokens
  CUDA_KERNEL_LOOP(i, data_dim * num_tokens * num_chosen_experts) {
    int token_index = i / (data_dim * num_chosen_experts);
    int chosen_exp_index = (i % (data_dim * num_chosen_experts)) / data_dim;
    int data_dim_index = (i % (data_dim * num_chosen_experts)) % data_dim;
    assert(i == data_dim * num_chosen_experts * token_index + data_dim * chosen_exp_index + data_dim_index);
    int j = data_dim * token_index + data_dim_index;
    tokens_array[i] = input[j];
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

  // int num_experts = m->num_experts;
  // int expert_start_index = m->experts_start_idx;
  // bool use_bias = m->use_bias;
  // ActiMode activation = m->activation;
  int data_dim = m->data_dim;
  int num_chosen_experts = m->num_chosen_experts;
  int num_tokens = m->effective_batch_size;

  assert(chosen_experts == num_chosen_experts);
  assert(num_tokens == batch_size);

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

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

  // // 1. port the 2d matrix into Thrust
  // int k = chosen_experts == 2 ? chosen_experts : 2;
  // thrust::device_vector< float > input_tokens1(input, input + input_volume);
  // thrust::device_vector< float > input_tokens2(input, input + input_volume);
  // thrust::device_vector< float > replicated_tokens(k*input_volume);

  // // 2. replicate the tokens (assuming k=2 here)
  // thrust::device_vector< int > sorting_keys1(input_volume);
  // thrust::sequence(sorting_keys1.begin(), sorting_keys1.end());

  // // divide sorting_keys1 by data_dim and then multiply by k
  // thrust::transform(sorting_keys1.begin(), sorting_keys1.end(),
  // thrust::make_constant_iterator(data_dim), sorting_keys1.begin(),
  // divide_functor()); thrust::transform(sorting_keys1.begin(),
  // sorting_keys1.end(), thrust::make_constant_iterator(k),
  // sorting_keys1.begin(), multiply_functor());

  // // obtain the i-th sorting keys by adding +1 to the i-1 sorting keys
  // thrust::device_vector< int > sorting_keys2(input_volume);
  // thrust::copy(sorting_keys1.begin(), sorting_keys1.end(),
  // sorting_keys2.begin()); thrust::transform(sorting_keys1.begin(),
  // sorting_keys1.end(), thrust::make_constant_iterator(1),
  // sorting_keys1.begin(), thrust::plus<int>());

  // // populate the replicated_tokens vector with k side-by-side copies of each
  // token thrust::device_vector< int > merged_keys(2*input_volume);
  // thrust::merge_by_key(sorting_keys1.begin(), sorting_keys1.end(),
  // sorting_keys2.begin(), sorting_keys2.end(), input_tokens1, input_tokens2,
  // merged_keys, replicated_tokens, thrust::less<int>());

  // // 3. sort the tokens by expert index to which they are assigned
  // thrust::device_vector< int > expert_assignments(indices, indices +
  // k*num_tokens); 
  //thrust::sort_by_key(expert_assignments, expert_assignments + k*num_tokens, replicated_tokens); // FIX: no operator "+" matches these
  // operands

  // // 4. matrix multiply each slice of min(expert_capacity,
  // end_of_expert_slice) tokens by the corresponding weight

  // // get list of experts (in this block) receiving non-zero tokens
  // thrust::device_vector< int > experts_in_use(k*num_tokens);
  // int tot_exps = thrust::unique_copy(expert_assignments.begin(),
  // expert_assignments.end(), experts_in_use) - experts_in_use.begin(); // FIX:
  // no operator "-" matches these operands struct is_expert_in_block {
  //   __host__ __device__
  //   bool operator()(const int x)
  //   {
  //     return x >= experts_start_idx && x < experts_start_idx+num_experts;
  //   }
  // };
  // int n_used_experts = thrust::remove_if(experts_in_use.begin(),
  // experts_in_use.begin() + tot_exps, is_expert_in_block()) -
  // experts_in_use.begin();

  // // TODO: pad replicated tokens array:

  // // get the indexes of each slice of tokens
  // thrust::device_vector< int > slice_indices(k*num_tokens);
  // thrust::sequence(slice_indices.begin(), slice_indices.end());
  // int num_slices = (thrust::unique_by_key(expert_assignments.begin(),
  // expert_assignments.end(), slice_indices.begin())).first -
  // expert_assignments.begin();

  int kernel1_parallelism = data_dim * num_tokens * num_chosen_experts;
  experts_forward_kernel1<<<GET_BLOCKS(kernel1_parallelism), min(CUDA_NUM_THREADS, (int)kernel1_parallelism), 0, stream>>>(data_dim, num_chosen_experts, num_tokens, m->dev_sorted_tokens, input);

  // sort tokens by key (where key is the indices with the assignment)
  // * create a vector of sequential indices (range (1... total_num_tokens)) : original_order
  // * sort the original_order vector by key, using the same key (indices) used to sort the tokens
  // -> sorting the tokens vector by the keys in original_order allows us to recover the original order

  
  // detecting indexes of slices (each slice is the group of tokens assigned to an expert)
  // -> compute number of tokens assigned to each expert : max_tokens
  // -> min(max_tokens, expert_capacity)
  // compute number of experts in block receiving non-zero number of tokens (non_zero_num_experts)
  
  // create new array, of size min(max_tokens, expert_capacity) * non_zero_num_experts. Initialize everything to 0.
  // - copy data from each slice's start_index to min(start_index+expert_capacity, slice end_index) to new array at location non_zero_expert_index * min(max_tokens, expert_capacity).
  // - create a new original_order_shortened vector where you delete the entries corresponding to tokens that are dropped because the expert capacity has been exceeded. 

  // cublas gemm batched <- pass to the operator the list of weights for non-zero experts
  
  // left todo:
  // remove padding
  //  -> keep track of how much padding was added to each slice. After gemm batched matmul, remove out_dim*padding_size from each slice to obtain end_index of each slice. Copy each slice of output to an array where you have sequential data
  // multiply by coefficients
    //  -> use original_order_shortened vector to reorder the output values according to the original token order 
    // figure out how to either remove coefficients that map to tokens that have been dropped, or how to add padding to the tokens (in the new order), so that the number of tokens matches the original batch size
  // using CUBLAS, multiply each output by the corresponding coefficient (now that outputs are in the same order as the coefficients) and sum the outputs for the same token

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
  checkCUDA(
      cudaMalloc(&dev_sorted_tokens,
                 data_dim * effective_batch_size * num_chosen_experts * sizeof(float)));
}
ExpertsMeta::~ExpertsMeta(void) {
  checkCUDA(cudaFree(&dev_sorted_tokens));
}

}; // namespace FlexFlow

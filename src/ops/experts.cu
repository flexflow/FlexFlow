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
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

namespace FlexFlow {

/**
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
**/

__global__ void experts_forward_prepare_kernel(int num_experts,
                                       int num_tokens,
                                       int num_chosen_experts,
                                       int out_dim,

                                       int const *indices,       // @In: Indices for chosen experts 
                                       
                                       float const **weights,    // @In: Experts' weights
                                       float **weight_idx_array, // @Out: Aarray for GemmBatchedEx 

                                       float const *input,       // @In: Tokens' values
                                       float **token_idx_arrary, // @Out: Barray for GemmBatchedEx 

                                       float const *gemm_results,       // @In: GemmBatchedEx's result tensor (out_dim, batch_size)
                                       float **result_idx_arrary // @Out: Carray for GemmBatchedEx
                                       ) {
    // Initialize Aarray and Barray for Cublas GemmBatchedEx
    CUDA_KERNEL_LOOP(i, num_tokens * num_chosen_experts) {
        int expert_index = indices[i];
        int token_index = i / num_chosen_experts;
        if (expert_index && expert_index < num_experts) {
            weight_idx_array[i] = const_cast<float*>(weights[expert_index]);
            token_idx_arrary[i] = const_cast<float*>(input + i * token_index * sizeof(float));
            result_idx_arrary[i] = const_cast<float*>(gemm_results + i * out_dim * sizeof(float));
        }
    }
}

void experts_forward_GemmBatched_kernel(cublasHandle_t const handle,
                                                   float **weight_ptr,
                                                   float **input_ptr,
                                                   float **output_ptr,
                                                   //  void const *bias_ptr,
                                                   int in_dim,
                                                   int out_dim,
                                                   int num_tokens,
                                                   int num_chosen_experts,
                                                   ffStream_t stream) {
  // checkCUDA(cublasSetStream(handle, stream));
  // checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;

  // cudaDataType_t input_type = ff_to_cuda_datatype(m->input_type);
  // cudaDataType_t weight_type = ff_to_cuda_datatype(m->weight_type);
  // cudaDataType_t output_type = ff_to_cuda_datatype(m->output_type);
  cudaDataType_t input_type = CUDA_R_32F;
  cudaDataType_t weight_type = CUDA_R_32F;
  cudaDataType_t output_type = CUDA_R_32F;

  cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;

  cublasGemmBatchedEx(handle,
                      CUBLAS_OP_T, // Tranpose Weight, shape (in_dim, out_dim) => (out_dim, in_dim) 
                      CUBLAS_OP_N, // Input_token, shape (in_dim, 1)
                      out_dim,     // num_row of (A, C) = out_dim
                      1,           // num_col of (B, C) = 1
                      in_dim,      // num_col of A and num_rows of B = in_dim
                      &alpha,
                      (const void**)weight_ptr,  // Aarray (num_tokens * chosen_experts, in_dim, out_dim)
                      weight_type,
                      in_dim,      // Leading Dimension of weight before transpose
                      (const void**)input_ptr,  // Barray (num_tokens * chosen_experts, in_dim, 1)
                      input_type,
                      in_dim,      // Leading Dimension of input_token
                      &beta,
                      (void**)output_ptr,  // Carray (num_tokens * chosen_experts, out_dim, 1)
                      output_type,
                      out_dim,     // Leading Dimension of output
                      num_tokens * num_chosen_experts,  // Total submatrixs
                      compute_type,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
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
//   int experts_start_idx = m->experts_start_idx;
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

  int prepare_kernel_parallelism = num_tokens * num_chosen_experts;
  experts_forward_prepare_kernel<<<GET_BLOCKS(prepare_kernel_parallelism),
                            min(CUDA_NUM_THREADS, (int)prepare_kernel_parallelism),
                            0,
                            stream>>>(num_experts_per_block,
                                      num_tokens,
                                      num_chosen_experts,
                                      out_dim,
                                      indices,
                                      weights,
                                      m->dev_weight_idx_array,   // @Out: Aarray for GemmBatchedEx                                       
                                      input,
                                      m->dev_token_idx_arrary,   // @Out: Barray for GemmBatchedEx 
                                      m->dev_gemm_result,
                                      m->dev_result_idx_array // @Out: Carray for GemmBatchedEx 
                                    );

  cublasHandle_t handle;
  cublasCreate(&handle);

  experts_forward_GemmBatched_kernel(
      handle,
      m->dev_weight_idx_array,
      m->dev_token_idx_arrary,
      m->dev_result_idx_array,
                                                      //  bias_ptr,
      data_dim,
      out_dim,
      num_tokens,
      num_chosen_experts,
      stream);

/**
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


  // thrust_tokens_ptr will contain the input tokens replicated k times and
  // sorted by expert assignment
  thrust::device_ptr<float> thrust_tokens_ptr =
      thrust::device_pointer_cast(m->dev_sorted_tokens);
  // thrust_indices_ptr contain the expert assignment indices, each replicated
  // data_dim times
  thrust::device_ptr<int> thrust_indices_ptr =
      thrust::device_pointer_cast(m->dev_replicated_indices);
  // sort thrust_tokens_ptr by key (expert assignments). Use stable sort to
  // avoid sorting values that don't have an order (we assign the same index to
  // all the data_dim floats that represent the same token and we want to make
  // sure that their order does not change during sorting)
  thrust::stable_sort_by_key(thrust::device,
                             thrust_indices_ptr,
                             thrust_indices_ptr +
                                 num_chosen_experts * num_tokens * data_dim,
                             thrust_tokens_ptr,
                             thrust::greater<int>());

  // given the list of indices (representing the token -> expert assignment) in
  // thrust_indices_ptr, which have been sorted together with the tokens in the
  // step above, get the index (i.e. the slot) in the sorted thrust_indices_ptr
  // where each distinct expert index appears for the first time. Save the
  // result in thrust_exp_slice_ptr. The result will consist of experts_num
  // integers, or less if some experts don't receive any token, so their indices
  // don't appear in the assignments
  thrust::device_ptr<int> thrust_exp_slice_ptr =
      thrust::device_pointer_cast(m->dev_exp_slice_indices);
  thrust::sequence(thrust_exp_slice_ptr,
                   thrust_exp_slice_ptr +
                       num_chosen_experts * num_tokens * data_dim);
  // non_zero_tokens_experts holds the number of experts receiving at least one
  // token
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

  cublasHandle_t handle;
  cublasCreate(&handle);

  int total_matrix = expert_capacity * num_experts;

  experts_forward_GemmBatched_kernel(
      handle,
      thrust::raw_pointer_cast(thrust_dev_tokens_in_use_ptr), // tokens
      m->dev_gemm_result,
      thrust::raw_pointer_cast(thrust_exp_slice_ptr), // experts
                                                      //  bias_ptr,
      data_dim,
      out_dim,
      batch_size,
      expert_capacity,
      num_experts,
      stream);
**/

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
                         int _out_dim,
                         int _effective_batch_size,
                         int _num_chosen_experts,
                         float _alpha,
                         bool _use_bias,
                         ActiMode _activation)
    : OpMeta(handler), num_experts(_num_experts),
      experts_start_idx(_experts_start_idx), data_dim(_data_dim),
      out_dim(_out_dim),
      effective_batch_size(_effective_batch_size),
      num_chosen_experts(_num_chosen_experts), alpha(_alpha),
      use_bias(_use_bias), activation(_activation) {
  expert_capacity =
      ceil(alpha * num_chosen_experts / num_experts * effective_batch_size);
//   checkCUDA(cudaMalloc(&dev_sorted_tokens,
//                        data_dim * effective_batch_size * num_chosen_experts *
//                            sizeof(float)));
//   checkCUDA(cudaMalloc(&dev_replicated_indices,
//                        data_dim * effective_batch_size * num_chosen_experts *
//                            sizeof(int)));
//   checkCUDA(cudaMalloc(&dev_exp_slice_indices, num_experts * sizeof(int)));
//   checkCUDA(
//       cudaMalloc(&dev_tokens_in_use,
//                  data_dim * expert_capacity * num_experts * sizeof(float)));

  checkCUDA(
    cudaMalloc(&dev_gemm_result,
                 out_dim * num_chosen_experts * effective_batch_size * sizeof(float)));
  checkCUDA(
    cudaMalloc(&dev_token_idx_arrary,
                 num_chosen_experts * effective_batch_size * sizeof(float*)));
  checkCUDA(
    cudaMalloc(&dev_weight_idx_array,
                 num_chosen_experts * effective_batch_size * sizeof(float*)));
  checkCUDA(
    cudaMalloc(&dev_result_idx_array,
                 num_chosen_experts * effective_batch_size * sizeof(float*)));       
}
ExpertsMeta::~ExpertsMeta(void) {
//   checkCUDA(cudaFree(&dev_sorted_tokens));
//   checkCUDA(cudaFree(&dev_replicated_indices));
//   checkCUDA(cudaFree(&dev_exp_slice_indices));
//   checkCUDA(cudaFree(&dev_tokens_in_use));

  checkCUDA(cudaFree(&dev_gemm_result));
  checkCUDA(cudaFree(&dev_token_idx_arrary));
  checkCUDA(cudaFree(&dev_weight_idx_array));
  checkCUDA(cudaFree(&dev_result_idx_array));
}

}; // namespace FlexFlow

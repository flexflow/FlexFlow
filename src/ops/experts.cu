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
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Thrust-related headers
#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT 1
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <chrono>
#include <thread>

namespace FlexFlow {

__global__ void experts_forward_prepare_kernel(
    int num_valid_assignments,
    int expert_capacity,
    int lb_index,
    int experts_start_idx,
    int num_experts_per_block,
    int num_chosen_experts,
    int data_dim,
    int out_dim,
    bool use_bias,
    int *sorted_indices,
    int *expert_start_indexes,
    int *exp_local_label_to_index,
    int *destination_start_indices,
    int *original_indices,
    float const *input, // @In: Tokens' values (in_dim, batch_size)
    float *output,
    float const **token_idx_array,  // @Out: Barray for GemmBatchedEx
    float const **weights,          // @In: Experts' weights
    float const **weight_idx_array, // @Out: Aarray for GemmBatchedEx
    float const **bias_idx_array,   // @Out: Experts' bias
    float const *coefficients,      // @In: topk_gate_predss coefficients tensor
                                    // (num_chosen_experts, batch_size)
    float const **coefficient_idx_array, // @Out: Barray for Aggregation
    float **output_idx_array) {

  CUDA_KERNEL_LOOP(i, num_valid_assignments) {
    int global_expert_label = sorted_indices[lb_index + i];
    assert(global_expert_label >= experts_start_idx &&
           global_expert_label < experts_start_idx + num_experts_per_block);
    int local_expert_label = global_expert_label - experts_start_idx;
    int expert_index = exp_local_label_to_index[local_expert_label];
    int within_expert_offset = i - expert_start_indexes[expert_index];
    if (within_expert_offset < expert_capacity) {
      int rev_idx = original_indices[i + lb_index];
      int token_idx = (rev_idx / num_chosen_experts);

      token_idx_array[destination_start_indices[expert_index] +
                      within_expert_offset] = &input[token_idx * data_dim];
      weight_idx_array[destination_start_indices[expert_index] +
                       within_expert_offset] =
          weights[local_expert_label * (1 + use_bias)];
      if (use_bias) {
        bias_idx_array[destination_start_indices[expert_index] +
                       within_expert_offset] =
            weights[local_expert_label * (1 + use_bias) + use_bias];
      }
      coefficient_idx_array[destination_start_indices[expert_index] +
                            within_expert_offset] = &coefficients[rev_idx];
      output_idx_array[destination_start_indices[expert_index] +
                       within_expert_offset] = &output[token_idx * out_dim];
    }
  }
}

bool use_activation(ActiMode mode) {
  switch (mode) {
    case AC_MODE_RELU:
    case AC_MODE_SIGMOID:
    case AC_MODE_TANH:
      return true;
    case AC_MODE_NONE:
      return false;
    default:
      assert(0);
      break;
  }
  return false;
}

void experts_forward_GemmBatched_kernel(ExpertsMeta const *m,
                                        void const **weights_ptr,
                                        void const **input_ptr,
                                        void **output_ptr,
                                        void const **bias_ptr,
                                        ActiMode activation,
                                        int in_dim,
                                        int out_dim,
                                        int num_tokens,
                                        int num_chosen_experts,
                                        int gemm_batch_count,
                                        ffStream_t stream) {

  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;

  // cudaDataType_t input_type = ff_to_cuda_datatype(m->input_type);
  // cudaDataType_t weight_type = ff_to_cuda_datatype(m->weight_type);
  // cudaDataType_t output_type = ff_to_cuda_datatype(m->output_type);
  cudaDataType_t input_type = CUDA_R_32F;
  cudaDataType_t weight_type = CUDA_R_32F;
  cudaDataType_t output_type = CUDA_R_32F;

  cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;

  checkCUDA(cublasGemmBatchedEx(
      m->handle.blas,
      CUBLAS_OP_T, // Tranpose Weight, shape (in_dim, out_dim) => (out_dim,
                   // in_dim)
      CUBLAS_OP_N, // Input_token, shape (in_dim, 1)
      out_dim,     // num_row of (A, C) = out_dim
      1,           // num_col of (B, C) = 1
      in_dim,      // num_col of A and num_rows of B = in_dim
      &alpha,
      weights_ptr, // Aarray (num_tokens * chosen_experts, in_dim, out_dim)
      weight_type,
      in_dim,    // Leading Dimension of weight before transpose
      input_ptr, // Barray (num_tokens * chosen_experts, in_dim, 1)
      input_type,
      in_dim, // Leading Dimension of input_token
      &beta,
      output_ptr, // Carray (num_tokens * chosen_experts, out_dim, 1)
      output_type,
      out_dim,          // Leading Dimension of output
      gemm_batch_count, // Total submatrixes
      compute_type,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // TODO 2: bias and activations
  if (m->use_bias) {
    checkCUDA(cublasGemmBatchedEx(
        m->handle.blas,
        CUBLAS_OP_T, // Bias, shape (out_dim, 1)
        CUBLAS_OP_N, // Coefficient, shape (1, 1)
        out_dim,     // num_row of (A, C) = out_dim
        1,           // num_col of (B, C) = 1
        1,           // num_col of A and num_rows of B = 1
        &alpha,
        bias_ptr, // bias tensor (out_dim, 1)
        weight_type,
        out_dim,                         // Leading Dimension of bias tensor
        (void const **)m->one_ptr_array, // all-one tensor (1, 1)
        CUDA_R_32F,
        1, // Leading Dimension of all-one tensor
        &alpha,
        output_ptr, // Carray (num_tokens * chosen_experts, out_dim, 1)
        output_type,
        out_dim,          // Leading Dimension of output
        gemm_batch_count, // Total submatrixs
        compute_type,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

  if (use_activation(activation)) {
    cudnnActivationMode_t mode;
    switch (activation) {
      case AC_MODE_RELU:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case AC_MODE_SIGMOID:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      default:
        // Unsupported activation mode
        assert(false);
    }
    checkCUDNN(cudnnSetActivationDescriptor(
        m->actiDesc, mode, CUDNN_PROPAGATE_NAN, 0.0));
    checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor,
                                          CUDNN_TENSOR_NCHW,
                                          // CUDNN_DATA_FLOAT,
                                          cuda_to_cudnn_datatype(output_type),
                                          gemm_batch_count,
                                          out_dim,
                                          1,
                                          1));
    checkCUDNN(cudnnActivationForward(m->handle.dnn,
                                      m->actiDesc,
                                      &alpha,
                                      m->outputTensor,
                                      m->batch_outputs[0],
                                      &beta,
                                      m->outputTensor,
                                      m->batch_outputs[0]));
  }
}

__global__ void experts_forward_aggregate_kernel(int num_tokens,
                                                 int gemm_batch_count,
                                                 int out_dim,
                                                 float *output,
                                                 float **results_ptr,
                                                 float const **coefficient_ptr,
                                                 float **output_ptr) {

  CUDA_KERNEL_LOOP(i, num_tokens * out_dim) {
    output[i] = 0.0f;
  }

  __syncthreads();

  CUDA_KERNEL_LOOP(i, gemm_batch_count * out_dim) {
    int token_index = i / out_dim;
    int emb_index = i % out_dim;
    float res =
        results_ptr[token_index][emb_index] * (*coefficient_ptr[token_index]);
    atomicAdd(output_ptr[token_index] + emb_index, res);
  }
}

struct exceeds_expert_capacity {
  int _expert_capacity;
  exceeds_expert_capacity(int expert_capacity)
      : _expert_capacity(expert_capacity){};
  __host__ __device__ bool operator()(int x) {
    return x > _expert_capacity;
  }
};

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

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  int num_experts_per_block = m->num_experts;
  int experts_start_idx = m->experts_start_idx;
  bool use_bias = m->use_bias;
  ActiMode activation = m->activation;
  int data_dim = m->data_dim;
  int num_chosen_experts = m->num_chosen_experts;
  int num_tokens = m->effective_batch_size;
  int expert_capacity = m->expert_capacity;

  assert(chosen_experts == num_chosen_experts);
  assert(num_tokens == batch_size);
  assert(out_dim == m->out_dim);

  // TODO: remove this once we condense all weights in a single tensor
  // currently each weight matrix is placed on GPU by Legion, but the array
  // holding the pointers to each weight matrix is on CPU
  cudaMemcpy(m->dev_weights,
             weights,
             num_experts_per_block * (1 + use_bias) * sizeof(float *),
             cudaMemcpyHostToDevice);

  int num_indices = num_tokens * num_chosen_experts;
  // sort the indices and coefficients by expert. Keep track of the original
  // position of each index/coefficient using the original_indices array
  thrust::device_ptr<int const> thrust_indices =
      thrust::device_pointer_cast(indices);
  thrust::device_ptr<int> sorted_indices =
      thrust::device_pointer_cast(m->sorted_indices);
  thrust::copy(thrust::device,
               thrust_indices,
               thrust_indices + num_indices,
               sorted_indices);

  thrust::device_ptr<int> original_indices =
      thrust::device_pointer_cast(m->original_indices);
  thrust::sequence(
      thrust::device, original_indices, original_indices + num_indices);

  thrust::stable_sort_by_key(thrust::device,
                             sorted_indices,
                             sorted_indices + num_indices,
                             original_indices);

  // get lower and upper bound of indices corresponding to experts in the block
  thrust::device_ptr<int> lb = thrust::lower_bound(
      sorted_indices, sorted_indices + num_indices, experts_start_idx);
  thrust::device_ptr<int> ub =
      thrust::upper_bound(sorted_indices,
                          sorted_indices + num_indices,
                          experts_start_idx + num_experts_per_block);

  int lb_index = lb - sorted_indices;
  int ub_index = ub - sorted_indices;
  int num_valid_assignments = ub_index - lb_index;
  if (num_valid_assignments == 0) {
    return;
  }

  // create "exp_local_label_to_index", a mapping from local expert label to its
  // non-zero expert index
  thrust::device_ptr<int> non_zero_expert_labels =
      thrust::device_pointer_cast(m->non_zero_expert_labels);
  thrust::device_ptr<int> non_zero_expert_labels_end =
      thrust::unique_copy(lb, ub, non_zero_expert_labels);
  int non_zero_experts_count =
      non_zero_expert_labels_end - non_zero_expert_labels;

  using namespace thrust::placeholders;
  thrust::for_each(thrust::device,
                   non_zero_expert_labels,
                   non_zero_expert_labels + non_zero_experts_count,
                   _1 -=
                   experts_start_idx); // convert global indexes to local ones

  thrust::device_ptr<int> temp_sequence =
      thrust::device_pointer_cast(m->temp_sequence);
  thrust::sequence(
      thrust::device, temp_sequence, temp_sequence + non_zero_experts_count);

  thrust::device_ptr<int> exp_local_label_to_index =
      thrust::device_pointer_cast(m->exp_local_label_to_index);
  thrust::scatter(thrust::device,
                  temp_sequence,
                  temp_sequence + non_zero_experts_count,
                  non_zero_expert_labels,
                  exp_local_label_to_index);

  // get local start index (within lower/upper bound) for each expert receiving
  // non-zero tokens
  thrust::device_ptr<int> expert_start_indexes =
      thrust::device_pointer_cast(m->expert_start_indexes);
  thrust::sequence(thrust::device,
                   expert_start_indexes,
                   expert_start_indexes + num_valid_assignments);
  int start_indexes = (thrust::unique_by_key_copy(thrust::device,
                                                  lb,
                                                  ub,
                                                  expert_start_indexes,
                                                  temp_sequence,
                                                  expert_start_indexes))
                          .first -
                      temp_sequence;
  assert(start_indexes == non_zero_experts_count);

  // append ub_index
  expert_start_indexes[start_indexes] = ub_index;

  // get number of token assignment to each expert
  thrust::device_ptr<int> num_assignments_per_expert =
      thrust::device_pointer_cast(m->num_assignments_per_expert);
  thrust::transform(thrust::device,
                    expert_start_indexes + 1,
                    expert_start_indexes + non_zero_experts_count + 1,
                    expert_start_indexes,
                    num_assignments_per_expert,
                    thrust::minus<int>());

  // build destination_start_index array, telling us the first slot that belongs
  // to each expert in the destination array (after factoring in expert
  // capacity)
  thrust::device_ptr<int> destination_start_indices =
      thrust::device_pointer_cast(m->destination_start_indices);
  thrust::replace_copy_if(thrust::device,
                          num_assignments_per_expert,
                          num_assignments_per_expert + non_zero_experts_count,
                          destination_start_indices,
                          exceeds_expert_capacity(expert_capacity),
                          expert_capacity);

  int gemm_batch_count =
      thrust::reduce(thrust::device,
                     destination_start_indices,
                     destination_start_indices + non_zero_experts_count);

  thrust::exclusive_scan(thrust::device,
                         destination_start_indices,
                         destination_start_indices + non_zero_experts_count,
                         destination_start_indices,
                         0);

  cudaDeviceSynchronize();

  experts_forward_prepare_kernel<<<GET_BLOCKS(num_valid_assignments),
                                   min(CUDA_NUM_THREADS,
                                       (int)num_valid_assignments),
                                   0,
                                   stream>>>(num_valid_assignments,
                                             expert_capacity,
                                             lb_index,
                                             experts_start_idx,
                                             num_experts_per_block,
                                             num_chosen_experts,
                                             data_dim,
                                             out_dim,
                                             use_bias,
                                             m->sorted_indices,
                                             m->expert_start_indexes,
                                             m->exp_local_label_to_index,
                                             m->destination_start_indices,
                                             m->original_indices,
                                             input,
                                             output,
                                             m->token_idx_array,
                                             m->dev_weights,
                                             m->weight_idx_array,
                                             m->bias_idx_array,
                                             topk_gate_preds,
                                             m->coefficient_idx_array,
                                             m->output_idx_array);

  cudaDeviceSynchronize();

  experts_forward_GemmBatched_kernel(m,
                                     (void const **)m->weight_idx_array,
                                     (void const **)m->token_idx_array,
                                     (void **)m->dev_batch_outputs,
                                     (void const **)m->bias_idx_array,
                                     activation,
                                     data_dim,
                                     out_dim,
                                     num_tokens,
                                     num_chosen_experts,
                                     gemm_batch_count,
                                     stream);
  cudaDeviceSynchronize();

  int aggregation_parallelism =
      std::max(num_tokens, gemm_batch_count) * out_dim;
  experts_forward_aggregate_kernel<<<GET_BLOCKS(aggregation_parallelism),
                                     min(CUDA_NUM_THREADS,
                                         (int)aggregation_parallelism),
                                     0,
                                     stream>>>(num_tokens,
                                               gemm_batch_count,
                                               out_dim,
                                               output,
                                               m->dev_batch_outputs,
                                               m->coefficient_idx_array,
                                               m->output_idx_array);

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
      out_dim(_out_dim), effective_batch_size(_effective_batch_size),
      num_chosen_experts(_num_chosen_experts), alpha(_alpha),
      use_bias(_use_bias), activation(_activation) {
  expert_capacity =
      ceil(alpha * num_chosen_experts / num_experts * effective_batch_size);

  checkCUDA(
      cudaMalloc(&sorted_indices,
                 num_chosen_experts * effective_batch_size * sizeof(int)));
  checkCUDA(
      cudaMalloc(&original_indices,
                 num_chosen_experts * effective_batch_size * sizeof(int)));
  checkCUDA(cudaMalloc(&non_zero_expert_labels, num_experts * sizeof(int)));
  checkCUDA(cudaMalloc(
      &temp_sequence,
      std::max(num_experts, num_chosen_experts * effective_batch_size) *
          sizeof(int)));
  checkCUDA(cudaMalloc(&exp_local_label_to_index, num_experts * sizeof(int)));
  // expert_start_indexes needs one more slot to save the upper bound index
  checkCUDA(cudaMalloc(&expert_start_indexes, (num_experts + 1) * sizeof(int)));
  checkCUDA(cudaMalloc(&num_assignments_per_expert, num_experts * sizeof(int)));
  checkCUDA(cudaMalloc(&destination_start_indices, num_experts * sizeof(int)));

  checkCUDA(
      cudaMalloc(&token_idx_array,
                 num_chosen_experts * effective_batch_size * sizeof(float *)));
  checkCUDA(
      cudaMalloc(&dev_weights, num_experts * (1 + use_bias) * sizeof(float *)));
  checkCUDA(
      cudaMalloc(&weight_idx_array,
                 num_chosen_experts * effective_batch_size * sizeof(float *)));
  checkCUDA(
      cudaMalloc(&bias_idx_array,
                 num_chosen_experts * effective_batch_size * sizeof(float *)));
  checkCUDA(
      cudaMalloc(&coefficient_idx_array,
                 num_chosen_experts * effective_batch_size * sizeof(float *)));
  checkCUDA(
      cudaMalloc(&output_idx_array,
                 num_chosen_experts * effective_batch_size * sizeof(float *)));
  batch_outputs = new float *[num_chosen_experts * effective_batch_size];
  for (int i = 0; i < num_chosen_experts * effective_batch_size; i++) {
    checkCUDA(cudaMalloc(&batch_outputs[i], out_dim * sizeof(float)));
  }
  checkCUDA(
      cudaMalloc(&dev_batch_outputs,
                 num_chosen_experts * effective_batch_size * sizeof(float *)));
  checkCUDA(
      cudaMemcpy(dev_batch_outputs,
                 batch_outputs,
                 num_chosen_experts * effective_batch_size * sizeof(float *),
                 cudaMemcpyHostToDevice));
  float *dram_one_ptr = (float *)malloc(sizeof(float) * 1);
  for (int i = 0; i < 1; i++) {
    dram_one_ptr[i] = 1.0f;
  }
  float *fb_one_ptr;
  checkCUDA(cudaMalloc(&fb_one_ptr, sizeof(float) * 1));
  checkCUDA(cudaMemcpy(
      fb_one_ptr, dram_one_ptr, sizeof(float) * 1, cudaMemcpyHostToDevice));
  one_ptr = (float const *)fb_one_ptr;
  checkCUDA(
      cudaMalloc(&one_ptr_array,
                 num_chosen_experts * effective_batch_size * sizeof(float *)));
  for (int i = 0; i < num_chosen_experts * effective_batch_size; i++) {
    checkCUDA(cudaMemcpy(&one_ptr_array[i],
                         &fb_one_ptr,
                         sizeof(float *),
                         cudaMemcpyHostToDevice));
  }
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
}
ExpertsMeta::~ExpertsMeta(void) {

  checkCUDA(cudaFree(&sorted_indices));
  checkCUDA(cudaFree(&original_indices));
  checkCUDA(cudaFree(&non_zero_expert_labels));
  checkCUDA(cudaFree(&temp_sequence));
  checkCUDA(cudaFree(&exp_local_label_to_index));
  checkCUDA(cudaFree(&expert_start_indexes));
  checkCUDA(cudaFree(&num_assignments_per_expert));
  checkCUDA(cudaFree(&destination_start_indices));
  checkCUDA(cudaFree(&token_idx_array));
  checkCUDA(cudaFree(&dev_weights));
  checkCUDA(cudaFree(&weight_idx_array));
  checkCUDA(cudaFree(&coefficient_idx_array));
  checkCUDA(cudaFree(&output_idx_array));
  checkCUDA(cudaFree(&dev_batch_outputs));
  checkCUDA(cudaFree(&bias_idx_array));
  checkCUDA(cudaFree(&one_ptr));
  checkCUDA(cudaFree(&one_ptr_array));
  for (int i = 0; i < num_chosen_experts * effective_batch_size; i++) {
    checkCUDA(cudaFree(&batch_outputs[i]));
  }
  delete[] batch_outputs;
  checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
  checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
}

}; // namespace FlexFlow

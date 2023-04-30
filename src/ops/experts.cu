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

struct exceeds_expert_capacity {
  int _expert_capacity;
  exceeds_expert_capacity(int expert_capacity)
      : _expert_capacity(expert_capacity){};
  __host__ __device__ bool operator()(int x) {
    return x > _expert_capacity;
  }
};

void experts_forward_thrust_wrapper(ExpertsMeta const *m,
                                    int const *indices,
                                    int num_indices,
                                    int experts_start_idx,
                                    int num_experts_per_block,
                                    int expert_capacity,
                                    int *lb_index,
                                    int *ub_index,
                                    int *num_valid_assignments,
                                    int *non_zero_experts_count,
                                    int *start_indexes,
                                    int *gemm_batch_count,
                                    ffStream_t stream) {
  // sort the indices and coefficients by expert. Keep track of the original
  // position of each index/coefficient using the original_indices array
  thrust::device_ptr<int const> thrust_indices =
      thrust::device_pointer_cast(indices);
  thrust::device_ptr<int> sorted_indices =
      thrust::device_pointer_cast(m->sorted_indices);
  thrust::copy(thrust::cuda::par.on(stream),
               thrust_indices,
               thrust_indices + num_indices,
               sorted_indices);

  thrust::device_ptr<int> original_indices =
      thrust::device_pointer_cast(m->original_indices);
  thrust::sequence(thrust::cuda::par.on(stream),
                   original_indices,
                   original_indices + num_indices);

  thrust::stable_sort_by_key(thrust::cuda::par.on(stream),
                             sorted_indices,
                             sorted_indices + num_indices,
                             original_indices);

  // get lower and upper bound of token->expert assignments corresponding to
  // experts in the block
  thrust::device_ptr<int> lb = thrust::lower_bound(thrust::cuda::par.on(stream),
                                                   sorted_indices,
                                                   sorted_indices + num_indices,
                                                   experts_start_idx);
  thrust::device_ptr<int> ub =
      thrust::upper_bound(thrust::cuda::par.on(stream),
                          sorted_indices,
                          sorted_indices + num_indices,
                          experts_start_idx + num_experts_per_block - 1);
  // lowest index in the sorted indices array corresponding to an expert within
  // the block
  *lb_index = lb - sorted_indices;
  // 1 + largest index in the sorted indices array corresponding to an expert
  // within the block
  *ub_index = ub - sorted_indices;
  *num_valid_assignments = (*ub_index) - (*lb_index);
  if ((*num_valid_assignments) == 0) {
    return;
  }

  thrust::device_ptr<int> non_zero_expert_labels =
      thrust::device_pointer_cast(m->non_zero_expert_labels);
  // non_zero_expert_labels: a list of global labels of the experts in this
  // block receiving nonzero tokens
  thrust::device_ptr<int> non_zero_expert_labels_end = thrust::unique_copy(
      thrust::cuda::par.on(stream), lb, ub, non_zero_expert_labels);
  // number of experts in this block receiving at least one token
  *non_zero_experts_count = non_zero_expert_labels_end - non_zero_expert_labels;

  using namespace thrust::placeholders;
  // convert global labels to local labelling (e.g. expert 65->index 65-64=1 in
  // block containing experts 64-96) by substracting the experts_start_idx,
  // inplace.
  thrust::for_each(thrust::cuda::par.on(stream),
                   non_zero_expert_labels,
                   non_zero_expert_labels + (*non_zero_experts_count),
                   _1 -= experts_start_idx);

  thrust::device_ptr<int> temp_sequence =
      thrust::device_pointer_cast(m->temp_sequence);
  thrust::sequence(thrust::cuda::par.on(stream),
                   temp_sequence,
                   temp_sequence + (*non_zero_experts_count));

  // create "exp_local_label_to_index", a mapping from local expert label to its
  // non-zero expert index (i.e. expert with index i is the i-th expert in the
  // block to receive at least 1 token)
  thrust::device_ptr<int> exp_local_label_to_index =
      thrust::device_pointer_cast(m->exp_local_label_to_index);
  thrust::scatter(thrust::cuda::par.on(stream),
                  temp_sequence,
                  temp_sequence + (*non_zero_experts_count),
                  non_zero_expert_labels,
                  exp_local_label_to_index);

  // get local start index (within lower/upper bound) for each expert receiving
  // non-zero tokens
  thrust::device_ptr<int> expert_start_indexes =
      thrust::device_pointer_cast(m->expert_start_indexes);
  thrust::sequence(thrust::cuda::par.on(stream),
                   expert_start_indexes,
                   expert_start_indexes + (*num_valid_assignments));
  *start_indexes = (thrust::unique_by_key_copy(thrust::cuda::par.on(stream),
                                               lb,
                                               ub,
                                               expert_start_indexes,
                                               temp_sequence,
                                               expert_start_indexes))
                       .first -
                   temp_sequence;
  assert((*start_indexes) == (*non_zero_experts_count));

  // append ub_index
  expert_start_indexes[(*start_indexes)] = (*ub_index) - (*lb_index);

  // get number of token assignment to each expert
  thrust::device_ptr<int> num_assignments_per_expert =
      thrust::device_pointer_cast(m->num_assignments_per_expert);
  thrust::transform(thrust::cuda::par.on(stream),
                    expert_start_indexes + 1,
                    expert_start_indexes + (*non_zero_experts_count) + 1,
                    expert_start_indexes,
                    num_assignments_per_expert,
                    thrust::minus<int>());

  // build destination_start_index array, telling us the first slot that belongs
  // to each expert in the destination array (after factoring in expert
  // capacity)
  thrust::device_ptr<int> destination_start_indices =
      thrust::device_pointer_cast(m->destination_start_indices);
  thrust::replace_copy_if(thrust::cuda::par.on(stream),
                          num_assignments_per_expert,
                          num_assignments_per_expert +
                              (*non_zero_experts_count),
                          destination_start_indices,
                          exceeds_expert_capacity(expert_capacity),
                          expert_capacity);

  *gemm_batch_count =
      thrust::reduce(thrust::cuda::par.on(stream),
                     destination_start_indices,
                     destination_start_indices + (*non_zero_experts_count));

  thrust::exclusive_scan(thrust::cuda::par.on(stream),
                         destination_start_indices,
                         destination_start_indices + (*non_zero_experts_count),
                         destination_start_indices,
                         0);
}

__global__ void experts_forward_prepare_kernel(
    int num_valid_assignments,
    int expert_capacity,
    int lb_index,
    int experts_start_idx,
    int num_experts_per_block,
    int num_chosen_experts,
    int data_dim,
    int out_dim,
    int experts_num_layers,
    int experts_internal_dim_size,
    bool use_bias,
    int *sorted_indices,
    int *expert_start_indexes,
    int *exp_local_label_to_index,
    int *destination_start_indices,
    int *original_indices,
    float const *input, // @In: Tokens' values (in_dim, batch_size)
    float *output,
    float const **token_idx_array,   // @Out: Barray for GemmBatchedEx
    float const *weights,            // @In: Experts' weights
    float const *biases,             // @In: Experts' biases
    float const **weight_idx_array1, // @Out: Aarray for GemmBatchedEx
    float const **weight_idx_array2,
    float const **bias_idx_array1, // @Out: Experts' bias
    float const **bias_idx_array2,
    float const *coefficients, // @In: topk_gate_predss coefficients tensor
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
    int weight_params_count =
        experts_num_layers == 1
            ? data_dim * out_dim
            : experts_internal_dim_size * (data_dim + out_dim);
    if (within_expert_offset < expert_capacity) {
      int rev_idx = original_indices[i + lb_index];
      int token_idx = (rev_idx / num_chosen_experts);

      token_idx_array[destination_start_indices[expert_index] +
                      within_expert_offset] = &input[token_idx * data_dim];
      weight_idx_array1[destination_start_indices[expert_index] +
                        within_expert_offset] =
          &weights[local_expert_label * weight_params_count];
      if (experts_num_layers == 2) {
        weight_idx_array2[destination_start_indices[expert_index] +
                          within_expert_offset] =
            &weights[local_expert_label * weight_params_count +
                     (data_dim * experts_internal_dim_size)];
      }
      if (use_bias) {
        int bias_params_count = (experts_num_layers == 1)
                                    ? out_dim
                                    : (experts_internal_dim_size + out_dim);
        bias_idx_array1[destination_start_indices[expert_index] +
                        within_expert_offset] =
            &biases[local_expert_label * bias_params_count];
        if (experts_num_layers == 2) {
          bias_idx_array2[destination_start_indices[expert_index] +
                          within_expert_offset] =
              &biases[local_expert_label * bias_params_count +
                      experts_internal_dim_size];
        }
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
                                        void const **weights_ptr1,
                                        void const **weights_ptr2,
                                        void const **input_ptr,
                                        void **results_ptr1,
                                        void **results_ptr2,
                                        void const **bias_ptr1,
                                        void const **bias_ptr2,
                                        ActiMode activation,
                                        int in_dim,
                                        int out_dim,
                                        int experts_num_layers,
                                        int experts_internal_dim_size,
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

  int m_ = out_dim;
  int n = 1;
  int k = in_dim;
  void const **A = weights_ptr1;
  void const **B = input_ptr;
  void **C = results_ptr1;
  int lda = in_dim;
  int ldb = in_dim;
  int ldc = out_dim;
  if (experts_num_layers == 2) {
    m_ = ldc = experts_internal_dim_size;
  }
  checkCUDA(cublasGemmBatchedEx(
      m->handle.blas,
      CUBLAS_OP_T, // Tranpose Weight, shape (in_dim, out_dim) => (out_dim,
                   // in_dim)
      CUBLAS_OP_N, // Input_token, shape (in_dim, 1)
      m_,          // num_row of (A, C) = out_dim
      n,           // num_col of (B, C) = 1
      k,           // num_col of A and num_rows of B = in_dim
      &alpha,
      A, // Aarray (num_tokens * chosen_experts, in_dim, out_dim)
      weight_type,
      lda, // Leading Dimension of weight before transpose
      B,   // Barray (num_tokens * chosen_experts, in_dim, 1)
      input_type,
      ldb, // Leading Dimension of input_token
      &beta,
      C, // Carray (num_tokens * chosen_experts, out_dim, 1)
      output_type,
      ldc,              // Leading Dimension of output
      gemm_batch_count, // Total submatrixes
      compute_type,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  if (m->use_bias) {
    m_ = out_dim;
    n = 1;
    k = 1;
    A = bias_ptr1;
    B = (void const **)m->one_ptr_array;
    C = results_ptr1;
    lda = out_dim;
    ldb = 1;
    ldc = out_dim;
    if (experts_num_layers == 2) {
      m_ = lda = ldc = experts_internal_dim_size;
    }
    alpha = 1.0f, beta = 0.0f;
    checkCUDA(cublasGemmBatchedEx(
        m->handle.blas,
        CUBLAS_OP_N, // Bias, shape (out_dim, 1)
        CUBLAS_OP_N, // Coefficient, shape (1, 1)
        m_,          // num_row of (A, C) = out_dim
        n,           // num_col of (B, C) = 1
        k,           // num_col of A and num_rows of B = 1
        &alpha,
        A, // bias tensor (out_dim, 1)
        weight_type,
        lda, // Leading Dimension of bias tensor
        B,   // all-one tensor (1, 1)
        CUDA_R_32F,
        ldb, // Leading Dimension of all-one tensor
        &alpha,
        C, // Carray (num_tokens * chosen_experts, out_dim, 1)
        output_type,
        ldc,              // Leading Dimension of output
        gemm_batch_count, // Total submatrixs
        compute_type,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

  if (use_activation(activation)) {
    alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnActivationForward(m->handle.dnn,
                                      m->actiDesc,
                                      &alpha,
                                      m->resultTensorDesc1,
                                      m->batch_outputs1[0],
                                      &beta,
                                      m->resultTensorDesc1,
                                      m->batch_outputs1[0]));
  }

  if (experts_num_layers == 2) {
    m_ = out_dim;
    n = 1;
    k = experts_internal_dim_size;
    A = weights_ptr2;
    B = (void const **)results_ptr1;
    C = results_ptr2;
    lda = experts_internal_dim_size;
    ldb = experts_internal_dim_size;
    ldc = out_dim;
    alpha = 1.0f, beta = 0.0f;
    checkCUDA(cublasGemmBatchedEx(
        m->handle.blas,
        CUBLAS_OP_T, // Tranpose Weight, shape (in_dim, out_dim) => (out_dim,
                     // in_dim)
        CUBLAS_OP_N, // Input_token, shape (in_dim, 1)
        m_,          // num_row of (A, C) = out_dim
        n,           // num_col of (B, C) = 1
        k,           // num_col of A and num_rows of B = in_dim
        &alpha,
        A, // Aarray (num_tokens * chosen_experts, in_dim, out_dim)
        weight_type,
        lda, // Leading Dimension of weight before transpose
        B,   // Barray (num_tokens * chosen_experts, in_dim, 1)
        input_type,
        ldb, // Leading Dimension of input_token
        &beta,
        C, // Carray (num_tokens * chosen_experts, out_dim, 1)
        output_type,
        ldc,              // Leading Dimension of output
        gemm_batch_count, // Total submatrixes
        compute_type,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    if (m->use_bias) {
      m_ = out_dim;
      n = 1;
      k = 1;
      A = bias_ptr2;
      B = (void const **)m->one_ptr_array;
      C = results_ptr2;
      lda = out_dim;
      ldb = 1;
      ldc = out_dim;
      alpha = 1.0f, beta = 0.0f;
      checkCUDA(cublasGemmBatchedEx(
          m->handle.blas,
          CUBLAS_OP_N, // Bias, shape (out_dim, 1)
          CUBLAS_OP_N, // Coefficient, shape (1, 1)
          m_,          // num_row of (A, C) = out_dim
          n,           // num_col of (B, C) = 1
          k,           // num_col of A and num_rows of B = 1
          &alpha,
          A, // bias tensor (out_dim, 1)
          weight_type,
          lda, // Leading Dimension of bias tensor
          B,   // all-one tensor (1, 1)
          CUDA_R_32F,
          ldb, // Leading Dimension of all-one tensor
          &alpha,
          C, // Carray (num_tokens * chosen_experts, out_dim, 1)
          output_type,
          ldc,              // Leading Dimension of output
          gemm_batch_count, // Total submatrixs
          compute_type,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    if (use_activation(activation)) {
      alpha = 1.0f, beta = 0.0f;
      checkCUDNN(cudnnActivationForward(m->handle.dnn,
                                        m->actiDesc,
                                        &alpha,
                                        m->resultTensorDesc2,
                                        m->batch_outputs2[0],
                                        &beta,
                                        m->resultTensorDesc2,
                                        m->batch_outputs2[0]));
    }
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

/*static*/
void Experts::forward_kernel_wrapper(ExpertsMeta const *m,
                                     float const *input,
                                     int const *indices,
                                     float const *topk_gate_preds,
                                     float *output,
                                     float const *weights,
                                     float const *biases,
                                     int num_active_tokens,
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

  assert(num_active_tokens > 0);
  assert(num_active_tokens <= m->effective_batch_size);
  assert(m->effective_batch_size == batch_size);

  int num_experts_per_block = m->num_experts;
  int experts_start_idx = m->experts_start_idx;
  bool use_bias = m->use_bias;
  ActiMode activation = m->activation;
  int data_dim = m->data_dim;
  int num_chosen_experts = m->num_chosen_experts;
  // int num_tokens = m->effective_batch_size;
  int num_tokens = num_active_tokens;
  int expert_capacity = m->expert_capacity;

  assert(chosen_experts == num_chosen_experts);
  // assert(num_tokens == batch_size);
  assert(out_dim == m->out_dim);

  assert(weights != nullptr);
  assert(use_bias == (biases != nullptr));

  int num_indices = num_tokens * num_chosen_experts;
  // values below are set by Thrust in the experts_forward_thrust_wrapper
  // function
  int lb_index = 0;
  int ub_index = 0;
  int num_valid_assignments = 0;
  int non_zero_experts_count = 0;
  int start_indexes = 0;
  int gemm_batch_count = 0;

  experts_forward_thrust_wrapper(m,
                                 indices,
                                 num_indices,
                                 experts_start_idx,
                                 num_experts_per_block,
                                 expert_capacity,
                                 &lb_index,
                                 &ub_index,
                                 &num_valid_assignments,
                                 &non_zero_experts_count,
                                 &start_indexes,
                                 &gemm_batch_count,
                                 stream);

  // checkCUDA(cudaStreamSynchronize(stream));

#ifdef INFERENCE_TESTS
  // Checking
  // 1. check that m->sorted_indices contains indices sorted
  int *indices_cpu = download_tensor<int>(indices, num_indices);
  // assert(indices_cpu != nullptr);
  std::vector<int> indices_vec(indices_cpu, indices_cpu + num_indices);
  std::vector<int> indices_vec_sorted(indices_vec.size());
  std::copy(indices_vec.begin(), indices_vec.end(), indices_vec_sorted.begin());
  std::stable_sort(indices_vec_sorted.begin(), indices_vec_sorted.end());

  int *thrust_sorted_indices_cpu = download_tensor<int>(
      m->sorted_indices, m->num_chosen_experts * m->effective_batch_size);
  // assert(thrust_sorted_indices_cpu != nullptr);
  std::vector<int> thrust_sorted_indices_vec(
      thrust_sorted_indices_cpu, thrust_sorted_indices_cpu + num_indices);
  for (int i = 0; i < num_indices; i++) {
    if (indices_vec_sorted[i] != thrust_sorted_indices_vec[i]) {
      printf("i=%i\n", i);
      printf("indices: ");
      std::copy(indices_vec.begin(),
                indices_vec.end(),
                std::ostream_iterator<int>(std::cout, " "));
      std::cout << std::endl;
      printf("indices_vec_sorted: ");
      std::copy(indices_vec_sorted.begin(),
                indices_vec_sorted.end(),
                std::ostream_iterator<int>(std::cout, " "));
      std::cout << std::endl;
      printf("thrust_sorted_indices_vec: ");
      std::copy(thrust_sorted_indices_vec.begin(),
                thrust_sorted_indices_vec.end(),
                std::ostream_iterator<int>(std::cout, " "));
      std::cout << std::endl;
    }
    assert(indices_vec_sorted[i] == thrust_sorted_indices_vec[i]);
  }
  // 2. check that indices[m->original_indices[i]] = i
  int *thrust_original_indices_cpu = download_tensor<int>(
      m->original_indices, m->num_chosen_experts * m->effective_batch_size);
  // assert(thrust_original_indices_cpu != nullptr);
  std::vector<int> thrust_original_indices_vec(
      thrust_original_indices_cpu, thrust_original_indices_cpu + num_indices);
  for (int i = 0; i < num_indices; i++) {
    assert(indices_vec[thrust_original_indices_vec[i]] ==
           thrust_sorted_indices_vec[i]);
  }

  // 3. check that lb_index is the index of the first element greater or equal
  // to expert_start_idx
  // 4. check that ub_index is greater than last, or outside array
  std::vector<int>::iterator low, up;
  low = std::lower_bound(
      indices_vec_sorted.begin(), indices_vec_sorted.end(), experts_start_idx);
  up = std::upper_bound(indices_vec_sorted.begin(),
                        indices_vec_sorted.end(),
                        experts_start_idx + num_experts_per_block - 1);
  int lb_index_check = low - indices_vec_sorted.begin(),
      ub_index_check = up - indices_vec_sorted.begin();

  if (lb_index_check != lb_index || ub_index_check != ub_index) {
    printf("experts_start_idx: %i, num_experts_per_block: %i, lb_index: %i, "
           "lb_index_check: %i, ub_index: %i, ub_index_check: %i\n",
           experts_start_idx,
           num_experts_per_block,
           lb_index,
           lb_index_check,
           ub_index,
           ub_index_check);
    printf("indices_vec_sorted: ");
    std::copy(indices_vec_sorted.begin(),
              indices_vec_sorted.end(),
              std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
  }
  assert(lb_index_check == lb_index);
  assert(ub_index_check == ub_index);

  // 5. compute num_valid_assignments manually, and check that is equal to value
  // computed in thrust
  int num_valid_assignments_manual = ub_index_check - lb_index_check;
  assert(num_valid_assignments_manual == num_valid_assignments);

  // 6. check m->non_zero_expert_labels, *non_zero_experts_count
  std::set<int> non_zero_experts_check;
  for (int i = 0; i < num_indices; i++) {
    if (indices_vec_sorted[i] >= experts_start_idx &&
        indices_vec_sorted[i] < experts_start_idx + num_experts_per_block) {
      non_zero_experts_check.insert(indices_vec_sorted[i]);
    }
  }
  assert(non_zero_experts_count == non_zero_experts_check.size());
  // 7. check exp_local_label_to_index
  int *non_zero_expert_labels_cpu =
      download_tensor<int>(m->non_zero_expert_labels, non_zero_experts_count);
  // assert(non_zero_expert_labels_cpu != nullptr);
  std::vector<int> non_zero_expert_labels_vec(non_zero_expert_labels_cpu,
                                              non_zero_expert_labels_cpu +
                                                  non_zero_experts_count);
  assert(std::is_sorted(non_zero_expert_labels_vec.begin(),
                        non_zero_expert_labels_vec.end()));
  std::vector<int> non_zero_experts_check_vec;
  for (auto el : non_zero_experts_check) {
    non_zero_experts_check_vec.push_back(el - experts_start_idx);
  }
  assert(std::is_sorted(non_zero_experts_check_vec.begin(),
                        non_zero_experts_check_vec.end()));
  assert(non_zero_expert_labels_vec == non_zero_experts_check_vec);

  int *exp_local_label_to_index =
      download_tensor<int>(m->exp_local_label_to_index, non_zero_experts_count);
  // assert(exp_local_label_to_index != nullptr);
  std::vector<int> exp_local_label_to_index_vec(exp_local_label_to_index,
                                                exp_local_label_to_index +
                                                    non_zero_experts_count);
  int z = 0;
  for (int i = 0; i < non_zero_experts_count; i++) {
    if (non_zero_experts_check.find(i) != non_zero_experts_check.end()) {
      assert(exp_local_label_to_index_vec[i] == z);
      z++;
    }
  }

  // 8. Check expert_start_indexes
  int *expert_start_indices_thrust =
      download_tensor<int>(m->expert_start_indexes, non_zero_experts_count + 1);
  // assert(expert_start_indices_thrust != nullptr);
  std::vector<int> expert_start_indices_thrust_vec(
      expert_start_indices_thrust,
      expert_start_indices_thrust + non_zero_experts_count + 1);
  std::vector<int> expert_start_indices_cpu;
  std::set<int> exp_label;

  std::vector<int> num_assignments_per_expert_cpu;

  for (int i = lb_index; i < ub_index; i++) {
    assert(indices_vec_sorted[i] >= experts_start_idx &&
           indices_vec_sorted[i] < experts_start_idx + num_experts_per_block);
    if (exp_label.find(indices_vec_sorted[i]) == exp_label.end()) {
      exp_label.insert(indices_vec_sorted[i]);
      expert_start_indices_cpu.push_back(i - lb_index);

      num_assignments_per_expert_cpu.push_back(1);
    } else {
      num_assignments_per_expert_cpu[num_assignments_per_expert_cpu.size() -
                                     1] += 1;
    }
  }
  expert_start_indices_cpu.push_back(ub_index - lb_index);
  assert(num_assignments_per_expert_cpu.size() == non_zero_experts_count);
  /* std::cout << "indices_vec_sorted: ";
  for (int i=lb_index; i<ub_index; i++) {
    std::cout << indices_vec_sorted[i] << " ";
  }
  std::cout << "expert_start_indices_cpu: ";
  for (int i=0; i<expert_start_indices_cpu.size(); i++) {
    std::cout << expert_start_indices_cpu[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "expert_start_indices_thrust_vec: ";
  for (int i=0; i<expert_start_indices_thrust_vec.size(); i++) {
    std::cout << expert_start_indices_thrust_vec[i] << " ";
  }
  std::cout << std::endl; */
  assert(std::is_sorted(expert_start_indices_cpu.begin(),
                        expert_start_indices_cpu.end()));
  assert(expert_start_indices_cpu == expert_start_indices_thrust_vec);

  int *num_assignments_per_expert_thrust =
      (int *)calloc(non_zero_experts_count, sizeof(int));
  assert(num_assignments_per_expert_thrust != nullptr);
  assert(download_tensor<int>(m->num_assignments_per_expert,
                              num_assignments_per_expert_thrust,
                              non_zero_experts_count));
  assert(num_assignments_per_expert_thrust != nullptr);
  std::vector<int> num_assignments_per_expert_thrust_vec(
      num_assignments_per_expert_thrust,
      num_assignments_per_expert_thrust + non_zero_experts_count);
  assert(num_assignments_per_expert_cpu ==
         num_assignments_per_expert_thrust_vec);

  int *destination_start_indices_thrust =
      (int *)calloc(non_zero_experts_count, sizeof(int));
  assert(destination_start_indices_thrust != nullptr);
  assert(download_tensor<int>(m->destination_start_indices,
                              destination_start_indices_thrust,
                              non_zero_experts_count));
  assert(destination_start_indices_thrust != nullptr);
  std::vector<int> destination_start_indices_thrust_vec(
      destination_start_indices_thrust,
      destination_start_indices_thrust + non_zero_experts_count);
  std::vector<int> destination_start_indices_cpu;
  int gemm_batch_count_cpu = 0;
  for (int i = 0; i < num_assignments_per_expert_cpu.size(); i++) {
    if (i == 0) {
      destination_start_indices_cpu.push_back(0);
    } else {
      destination_start_indices_cpu.push_back(
          std::min(expert_capacity, num_assignments_per_expert_cpu[i - 1]));
    }
  }
  for (int i = 0; i < num_assignments_per_expert_cpu.size(); i++) {
    gemm_batch_count_cpu +=
        std::min(expert_capacity, num_assignments_per_expert_cpu[i]);
  }
  for (int i = 1; i < destination_start_indices_cpu.size(); i++) {
    destination_start_indices_cpu[i] += destination_start_indices_cpu[i - 1];
  }
  /*
  std::cout << "destination_start_indices_cpu: ";
  for (int i=0; i<destination_start_indices_cpu.size(); i++) {
    std::cout << destination_start_indices_cpu[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "destination_start_indices_thrust_vec: ";
  for (int i=0; i<destination_start_indices_thrust_vec.size(); i++) {
    std::cout << destination_start_indices_thrust_vec[i] << " ";
  }
  std::cout << std::endl; */
  assert(destination_start_indices_cpu == destination_start_indices_thrust_vec);
  assert(gemm_batch_count == gemm_batch_count_cpu);

  checkCUDA(cudaFreeHost(thrust_sorted_indices_cpu));
  checkCUDA(cudaFreeHost(thrust_original_indices_cpu));
  checkCUDA(cudaFreeHost(non_zero_expert_labels_cpu));
  checkCUDA(cudaFreeHost(exp_local_label_to_index));
  checkCUDA(cudaFreeHost(expert_start_indices_thrust));
  free(num_assignments_per_expert_thrust);
  free(destination_start_indices_thrust);

  non_zero_experts_check_vec.clear();
  non_zero_experts_check_vec.shrink_to_fit();
  expert_start_indices_cpu.clear();
  expert_start_indices_cpu.shrink_to_fit();
  destination_start_indices_cpu.clear();
  destination_start_indices_cpu.shrink_to_fit();
#endif

  assert(ub_index - lb_index == num_valid_assignments);
  assert(num_valid_assignments >= non_zero_experts_count);
  assert(non_zero_experts_count <= num_experts_per_block);
  if (non_zero_experts_count == 0) {
    assert(num_valid_assignments == 0 && gemm_batch_count == 0);
  } else {
    assert(num_valid_assignments > 0 && gemm_batch_count > 0);
  }
  assert(num_valid_assignments <= num_indices);
  assert(gemm_batch_count <= num_valid_assignments);

  if (num_valid_assignments == 0) {
    if (m->profiling) {
      cudaEventRecord(t_end, stream);
      cudaEventSynchronize(t_end);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, t_start, t_end);
      printf("forward_kernel_wrapper: %f ms\n", milliseconds);
    }
    return;
  }

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
                                             m->experts_num_layers,
                                             m->experts_internal_dim_size,
                                             use_bias,
                                             m->sorted_indices,
                                             m->expert_start_indexes,
                                             m->exp_local_label_to_index,
                                             m->destination_start_indices,
                                             m->original_indices,
                                             input,
                                             output,
                                             m->token_idx_array,
                                             weights,
                                             biases,
                                             m->weight_idx_array1,
                                             m->weight_idx_array2,
                                             m->bias_idx_array1,
                                             m->bias_idx_array2,
                                             topk_gate_preds,
                                             m->coefficient_idx_array,
                                             m->output_idx_array);

  // checkCUDA(cudaStreamSynchronize(stream));

#ifdef INFERENCE_TESTS
  std::vector<float const *> token_ptrs, weight_ptrs, bias_ptrs,
      coefficient_ptrs;
  std::vector<float *> output_ptrs;
  std::map<int, int> num_t_per_exp;
  for (int i = 0; i < num_indices; i++) {
    int global_exp_label = indices_vec[i];

    if (global_exp_label >= experts_start_idx &&
        global_exp_label < experts_start_idx + num_experts_per_block &&
        (num_t_per_exp.find(global_exp_label) == num_t_per_exp.end() ||
         num_t_per_exp[global_exp_label] < expert_capacity)) {
      if (num_t_per_exp.find(global_exp_label) == num_t_per_exp.end()) {
        num_t_per_exp[global_exp_label] = 1;
      } else {
        num_t_per_exp[global_exp_label] = num_t_per_exp[global_exp_label] + 1;
      }
      int token_idx = i / num_chosen_experts;
      // std::cout << "Push back token_idx (" << token_idx << ") * data_dim ("
      // << data_dim << "): " << token_idx*data_dim << std::endl;

      token_ptrs.push_back(&input[token_idx * data_dim]);
      coefficient_ptrs.push_back(&topk_gate_preds[i]);
      int local_exp_label = global_exp_label - experts_start_idx;
      weight_ptrs.push_back(&weights[local_exp_label * (out_dim * data_dim)]);
      output_ptrs.push_back(&output[token_idx * out_dim]);
      if (use_bias) {
        bias_ptrs.push_back(&biases[local_exp_label * out_dim]);
      }
    }
  }

  int i = 0, s = 0;
  for (auto it : num_t_per_exp) {
    int num_t = it.second;
    s += num_t;
    /* if (num_assignments_per_expert_cpu[i] != num_t) {
      std::cout << "num_assignments_per_expert_cpu: ";
      for (int j=0; j<num_assignments_per_expert_cpu.size(); j++) {
        std::cout << num_assignments_per_expert_cpu[j] << " ";
      }
      std::cout << std::endl;
      std::cout << "num_t_per_exp: ";
      for (auto it2 : num_t_per_exp) {
        std::cout << "(" << it2.first << ", " << it2.second << ") ";
      }
      std::cout << std::endl;
      std::cout << "expert capacity: " << expert_capacity << std::endl;
    }
    assert(num_assignments_per_expert_cpu[i] == num_t); */
    i++;
  }
  assert(s == gemm_batch_count);
  assert(token_ptrs.size() == gemm_batch_count &&
         weight_ptrs.size() == gemm_batch_count &&
         coefficient_ptrs.size() == gemm_batch_count &&
         output_ptrs.size() == gemm_batch_count);
  if (use_bias) {
    assert(bias_ptrs.size() == gemm_batch_count);
  }

  std::vector<float const *> token_ptrs_sorted(token_ptrs.size()),
      weight_ptrs_sorted(weight_ptrs.size()),
      bias_ptrs_sorted(bias_ptrs.size()),
      coefficient_ptrs_sorted(coefficient_ptrs.size());
  std::vector<float *> output_ptrs_sorted(output_ptrs.size());
  std::copy(token_ptrs.begin(), token_ptrs.end(), token_ptrs_sorted.begin());
  std::sort(token_ptrs_sorted.begin(), token_ptrs_sorted.end());
  std::copy(weight_ptrs.begin(), weight_ptrs.end(), weight_ptrs_sorted.begin());
  std::sort(weight_ptrs_sorted.begin(), weight_ptrs_sorted.end());
  std::copy(bias_ptrs.begin(), bias_ptrs.end(), bias_ptrs_sorted.begin());
  std::sort(bias_ptrs_sorted.begin(), bias_ptrs_sorted.end());
  std::copy(coefficient_ptrs.begin(),
            coefficient_ptrs.end(),
            coefficient_ptrs_sorted.begin());
  std::sort(coefficient_ptrs_sorted.begin(), coefficient_ptrs_sorted.end());
  std::copy(output_ptrs.begin(), output_ptrs.end(), output_ptrs_sorted.begin());
  std::sort(output_ptrs_sorted.begin(), output_ptrs_sorted.end());

  // Download
  float const **token_idx_array_thrust =
      (float const **)calloc(gemm_batch_count, sizeof(float const *));
  assert(token_idx_array_thrust);
  checkCUDA(cudaMemcpy(token_idx_array_thrust,
                       m->token_idx_array,
                       sizeof(float const *) * gemm_batch_count,
                       cudaMemcpyDeviceToHost));
  std::vector<float const *> token_idx_array_thrust_vec(
      token_idx_array_thrust, token_idx_array_thrust + gemm_batch_count);
  float const **weight_idx_array_thrust =
      (float const **)calloc(gemm_batch_count, sizeof(float const *));
  assert(weight_idx_array_thrust);
  checkCUDA(cudaMemcpy(weight_idx_array_thrust,
                       m->weight_idx_array1,
                       sizeof(float const *) * gemm_batch_count,
                       cudaMemcpyDeviceToHost));
  std::vector<float const *> weight_idx_array_thrust_vec(
      weight_idx_array_thrust, weight_idx_array_thrust + gemm_batch_count);
  float const **coefficient_idx_array_thrust =
      (float const **)calloc(gemm_batch_count, sizeof(float const *));
  assert(coefficient_idx_array_thrust);
  checkCUDA(cudaMemcpy(coefficient_idx_array_thrust,
                       m->coefficient_idx_array,
                       sizeof(float const *) * gemm_batch_count,
                       cudaMemcpyDeviceToHost));
  std::vector<float const *> coefficient_idx_array_thrust_vec(
      coefficient_idx_array_thrust,
      coefficient_idx_array_thrust + gemm_batch_count);
  float const **bias_idx_array_thrust =
      (float const **)calloc(gemm_batch_count, sizeof(float const *));
  assert(bias_idx_array_thrust);
  if (use_bias) {
    checkCUDA(cudaMemcpy(bias_idx_array_thrust,
                         m->bias_idx_array1,
                         sizeof(float const *) * gemm_batch_count,
                         cudaMemcpyDeviceToHost));
  }
  std::vector<float const *> bias_idx_array_thrust_vec(
      bias_idx_array_thrust, bias_idx_array_thrust + gemm_batch_count);
  float **output_idx_array_thrust =
      (float **)calloc(gemm_batch_count, sizeof(float *));
  assert(output_idx_array_thrust);
  checkCUDA(cudaMemcpy(output_idx_array_thrust,
                       m->output_idx_array,
                       sizeof(float *) * gemm_batch_count,
                       cudaMemcpyDeviceToHost));
  std::vector<float *> output_idx_array_thrust_vec(
      output_idx_array_thrust, output_idx_array_thrust + gemm_batch_count);

  std::vector<float const *> token_idx_array_thrust_vec_sorted(
      token_idx_array_thrust_vec.size()),
      weight_idx_array_thrust_vec_sorted(weight_idx_array_thrust_vec.size()),
      coefficient_idx_array_thrust_vec_sorted(
          coefficient_idx_array_thrust_vec.size()),
      bias_idx_array_thrust_vec_sorted(bias_idx_array_thrust_vec.size());
  std::vector<float *> output_idx_array_thrust_vec_sorted(
      output_idx_array_thrust_vec.size());
  std::copy(token_idx_array_thrust_vec.begin(),
            token_idx_array_thrust_vec.end(),
            token_idx_array_thrust_vec_sorted.begin());
  std::sort(token_idx_array_thrust_vec_sorted.begin(),
            token_idx_array_thrust_vec_sorted.end());
  std::copy(weight_idx_array_thrust_vec.begin(),
            weight_idx_array_thrust_vec.end(),
            weight_idx_array_thrust_vec_sorted.begin());
  std::sort(weight_idx_array_thrust_vec_sorted.begin(),
            weight_idx_array_thrust_vec_sorted.end());
  std::copy(coefficient_idx_array_thrust_vec.begin(),
            coefficient_idx_array_thrust_vec.end(),
            coefficient_idx_array_thrust_vec_sorted.begin());
  std::sort(coefficient_idx_array_thrust_vec_sorted.begin(),
            coefficient_idx_array_thrust_vec_sorted.end());
  std::copy(bias_idx_array_thrust_vec.begin(),
            bias_idx_array_thrust_vec.end(),
            bias_idx_array_thrust_vec_sorted.begin());
  std::sort(bias_idx_array_thrust_vec_sorted.begin(),
            bias_idx_array_thrust_vec_sorted.end());
  std::copy(output_idx_array_thrust_vec.begin(),
            output_idx_array_thrust_vec.end(),
            output_idx_array_thrust_vec_sorted.begin());
  std::sort(output_idx_array_thrust_vec_sorted.begin(),
            output_idx_array_thrust_vec_sorted.end());

  if (token_ptrs_sorted != token_idx_array_thrust_vec_sorted) {
    std::cout << "token_ptrs: ";
    for (int i = 0; i < token_ptrs_sorted.size(); i++) {
      std::cout << token_ptrs_sorted[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "token_idx_array_thrust_vec: ";
    for (int i = 0; i < token_idx_array_thrust_vec_sorted.size(); i++) {
      std::cout << token_idx_array_thrust_vec_sorted[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Input: " << input << std::endl;
    std::cout << "data_dim: " << data_dim << std::endl;
    std::cout << "out_dim: " << out_dim << std::endl;
    std::cout << "expert_start_idx: " << experts_start_idx << std::endl;
    std::cout << "indices: ";
    for (int i = 0; i < indices_vec.size(); i++) {
      std::cout << indices_vec[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "indices_vec_sorted: ";
    for (int i = 0; i < indices_vec_sorted.size(); i++) {
      std::cout << indices_vec_sorted[i] << " ";
    }
    std::cout << std::endl;
  }
  assert(token_ptrs_sorted == token_idx_array_thrust_vec_sorted);
  assert(weight_ptrs_sorted == weight_idx_array_thrust_vec_sorted);
  if (coefficient_ptrs_sorted != coefficient_idx_array_thrust_vec_sorted) {
    std::cout << "coefficient_ptrs_sorted: ";
    for (int i = 0; i < coefficient_ptrs_sorted.size(); i++) {
      std::cout << coefficient_ptrs_sorted[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "coefficient_idx_array_thrust_vec_sorted: ";
    for (int i = 0; i < coefficient_idx_array_thrust_vec_sorted.size(); i++) {
      std::cout << coefficient_idx_array_thrust_vec_sorted[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "topk_gate_preds: " << topk_gate_preds << std::endl;
    std::cout << "data_dim: " << data_dim << std::endl;
    std::cout << "out_dim: " << out_dim << std::endl;
    std::cout << "expert_start_idx: " << experts_start_idx << std::endl;
    std::cout << "indices: ";
    for (int i = 0; i < indices_vec.size(); i++) {
      std::cout << indices_vec[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "indices_vec_sorted: ";
    for (int i = 0; i < indices_vec_sorted.size(); i++) {
      std::cout << indices_vec_sorted[i] << " ";
    }
    std::cout << std::endl;
  }
  assert(coefficient_ptrs_sorted == coefficient_idx_array_thrust_vec_sorted);
  if (use_bias) {
    assert(bias_ptrs_sorted == bias_idx_array_thrust_vec_sorted);
  }
  assert(output_ptrs_sorted == output_idx_array_thrust_vec_sorted);

  assert(token_ptrs_sorted.size() == gemm_batch_count &&
         weight_ptrs_sorted.size() == gemm_batch_count &&
         coefficient_ptrs_sorted.size() == gemm_batch_count &&
         (!use_bias || bias_ptrs_sorted.size() == gemm_batch_count) &&
         output_ptrs_sorted.size() == gemm_batch_count);

  for (int i = 0; i < token_ptrs_sorted.size(); i++) {
    assert(token_ptrs_sorted[i]);
    assert(weight_ptrs_sorted[i]);
    assert(coefficient_ptrs_sorted[i]);
    if (use_bias) {
      assert(bias_ptrs_sorted[i]);
    }
    assert(output_ptrs_sorted[i]);
  }

  free(token_idx_array_thrust);
  free(weight_idx_array_thrust);
  free(coefficient_idx_array_thrust);
  free(bias_idx_array_thrust);
  free(output_idx_array_thrust);

  checkCUDA(cudaFreeHost(indices_cpu));
  indices_vec.clear();
  indices_vec.shrink_to_fit();
  indices_vec_sorted.clear();
  indices_vec_sorted.shrink_to_fit();
  num_assignments_per_expert_cpu.clear();
  num_assignments_per_expert_cpu.shrink_to_fit();

  token_ptrs.clear();
  token_ptrs.shrink_to_fit();
  token_ptrs_sorted.clear();
  token_ptrs_sorted.shrink_to_fit();
  weight_ptrs.clear();
  weight_ptrs.shrink_to_fit();
  weight_ptrs_sorted.clear();
  weight_ptrs_sorted.shrink_to_fit();
  bias_ptrs.clear();
  bias_ptrs.shrink_to_fit();
  bias_ptrs_sorted.clear();
  bias_ptrs_sorted.shrink_to_fit();
  coefficient_ptrs.clear();
  coefficient_ptrs.shrink_to_fit();
  output_ptrs.clear();
  output_ptrs.shrink_to_fit();
  output_ptrs_sorted.clear();
  output_ptrs_sorted.shrink_to_fit();

  token_idx_array_thrust_vec_sorted.clear();
  token_idx_array_thrust_vec_sorted.shrink_to_fit();
  weight_idx_array_thrust_vec_sorted.clear();
  weight_idx_array_thrust_vec_sorted.shrink_to_fit();
  coefficient_idx_array_thrust_vec_sorted.clear();
  coefficient_idx_array_thrust_vec_sorted.shrink_to_fit();
  bias_idx_array_thrust_vec_sorted.clear();
  bias_idx_array_thrust_vec_sorted.shrink_to_fit();
  output_idx_array_thrust_vec_sorted.clear();
  output_idx_array_thrust_vec_sorted.shrink_to_fit();

  // Check batch output pointers
  assert(gemm_batch_count <= m->effective_batch_size);
  float **dev_batch_outputs_cuda = (float **)calloc(
      num_chosen_experts * m->effective_batch_size, sizeof(float *));
  assert(dev_batch_outputs_cuda);
  checkCUDA(
      cudaMemcpy(dev_batch_outputs_cuda,
                 m->dev_batch_outputs1,
                 sizeof(float *) * num_chosen_experts * m->effective_batch_size,
                 cudaMemcpyDeviceToHost));
  std::vector<float *> dev_batch_outputs_cuda_vec(
      dev_batch_outputs_cuda,
      dev_batch_outputs_cuda + num_chosen_experts * m->effective_batch_size);

  std::vector<float *> batch_outputs_host_vec(
      m->batch_outputs1,
      m->batch_outputs1 + num_chosen_experts * m->effective_batch_size);
  assert(batch_outputs_host_vec == dev_batch_outputs_cuda_vec);

  /* std::cout << "dev_batch_outputs_cuda_vec[i]: ";
  for (int i=0; i<dev_batch_outputs_cuda_vec.size(); i++) {
    assert(dev_batch_outputs_cuda_vec[i]);
    if (i>0) {
      assert(dev_batch_outputs_cuda_vec[i] == dev_batch_outputs_cuda_vec[i-1] +
  out_dim);
    }
    std::cout << dev_batch_outputs_cuda_vec[i] << " ";
  }
  std::cout << std::endl; */

  free(dev_batch_outputs_cuda);
#endif

  experts_forward_GemmBatched_kernel(m,
                                     (void const **)m->weight_idx_array1,
                                     (void const **)m->weight_idx_array2,
                                     (void const **)m->token_idx_array,
                                     (void **)m->dev_batch_outputs1,
                                     (void **)m->dev_batch_outputs2,
                                     (void const **)m->bias_idx_array1,
                                     (void const **)m->bias_idx_array2,
                                     activation,
                                     data_dim,
                                     out_dim,
                                     m->experts_num_layers,
                                     m->experts_internal_dim_size,
                                     num_tokens,
                                     num_chosen_experts,
                                     gemm_batch_count,
                                     stream);

  // checkCUDA(cudaStreamSynchronize(stream));

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
                                               m->experts_num_layers == 1
                                                   ? m->dev_batch_outputs1
                                                   : m->dev_batch_outputs2,
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
                         int _experts_num_layers,
                         int _experts_internal_dim_size,
                         int _effective_batch_size,
                         int _num_chosen_experts,
                         float _alpha,
                         bool _use_bias,
                         ActiMode _activation)
    : OpMeta(handler), num_experts(_num_experts),
      experts_start_idx(_experts_start_idx), data_dim(_data_dim),
      out_dim(_out_dim), experts_num_layers(_experts_num_layers),
      experts_internal_dim_size(_experts_internal_dim_size),
      effective_batch_size(_effective_batch_size),
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
  // expert_start_indexes needs one more slot to save the upper bound index.
  // Initial sequence can require more space, though.
  checkCUDA(cudaMalloc(
      &expert_start_indexes,
      std::max(num_experts + 1, num_chosen_experts * effective_batch_size) *
          sizeof(int)));
  checkCUDA(cudaMalloc(&num_assignments_per_expert, num_experts * sizeof(int)));
  checkCUDA(cudaMalloc(&destination_start_indices, num_experts * sizeof(int)));

  checkCUDA(
      cudaMalloc(&token_idx_array,
                 num_chosen_experts * effective_batch_size * sizeof(float *)));
  checkCUDA(
      cudaMalloc(&weight_idx_array1,
                 num_chosen_experts * effective_batch_size * sizeof(float *)));
  checkCUDA(
      cudaMalloc(&bias_idx_array1,
                 num_chosen_experts * effective_batch_size * sizeof(float *)));
  checkCUDA(
      cudaMalloc(&coefficient_idx_array,
                 num_chosen_experts * effective_batch_size * sizeof(float *)));
  checkCUDA(
      cudaMalloc(&output_idx_array,
                 num_chosen_experts * effective_batch_size * sizeof(float *)));
  batch_outputs1 = new float *[num_chosen_experts * effective_batch_size];
  int batch_outputs1_dim =
      (experts_num_layers == 1) ? out_dim : experts_internal_dim_size;
  checkCUDA(cudaMalloc(&batch_outputs1[0],
                       batch_outputs1_dim * num_chosen_experts *
                           effective_batch_size * sizeof(float)));
  checkCUDA(cudaMemset(batch_outputs1[0],
                       0,
                       batch_outputs1_dim * num_chosen_experts *
                           effective_batch_size * sizeof(float)));
  for (int i = 1; i < num_chosen_experts * effective_batch_size; i++) {
    batch_outputs1[i] = batch_outputs1[i - 1] + batch_outputs1_dim;
  }
  checkCUDA(
      cudaMalloc(&dev_batch_outputs1,
                 num_chosen_experts * effective_batch_size * sizeof(float *)));
  checkCUDA(
      cudaMemcpy(dev_batch_outputs1,
                 batch_outputs1,
                 num_chosen_experts * effective_batch_size * sizeof(float *),
                 cudaMemcpyHostToDevice));
  if (experts_num_layers == 2) {
    checkCUDA(cudaMalloc(&weight_idx_array2,
                         num_chosen_experts * effective_batch_size *
                             sizeof(float *)));
    checkCUDA(cudaMalloc(&bias_idx_array2,
                         num_chosen_experts * effective_batch_size *
                             sizeof(float *)));
    batch_outputs2 = new float *[num_chosen_experts * effective_batch_size];
    checkCUDA(cudaMalloc(&batch_outputs2[0],
                         out_dim * num_chosen_experts * effective_batch_size *
                             sizeof(float)));
    checkCUDA(cudaMemset(batch_outputs2[0],
                         0,
                         out_dim * num_chosen_experts * effective_batch_size *
                             sizeof(float)));
    for (int i = 1; i < num_chosen_experts * effective_batch_size; i++) {
      batch_outputs2[i] = batch_outputs2[i - 1] + out_dim;
    }
    checkCUDA(cudaMalloc(&dev_batch_outputs2,
                         num_chosen_experts * effective_batch_size *
                             sizeof(float *)));
    checkCUDA(
        cudaMemcpy(dev_batch_outputs2,
                   batch_outputs2,
                   num_chosen_experts * effective_batch_size * sizeof(float *),
                   cudaMemcpyHostToDevice));
  }
  // Bias
  float *dram_one_ptr = (float *)malloc(sizeof(float) * 1);
  for (int i = 0; i < 1; i++) {
    dram_one_ptr[i] = 1.0f;
  }
  float *fb_one_ptr;
  checkCUDA(cudaMalloc(&fb_one_ptr, sizeof(float) * 1));
  checkCUDA(cudaMemcpy(
      fb_one_ptr, dram_one_ptr, sizeof(float) * 1, cudaMemcpyHostToDevice));
  one_ptr = (float const *)fb_one_ptr;
  free((void *)dram_one_ptr);
  checkCUDA(
      cudaMalloc(&one_ptr_array,
                 num_chosen_experts * effective_batch_size * sizeof(float *)));
  for (int i = 0; i < num_chosen_experts * effective_batch_size; i++) {
    checkCUDA(cudaMemcpy(&one_ptr_array[i],
                         &fb_one_ptr,
                         sizeof(float *),
                         cudaMemcpyHostToDevice));
  }
  // Activation
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
  checkCUDNN(cudnnCreateTensorDescriptor(&resultTensorDesc1));
  if (experts_num_layers == 2) {
    checkCUDNN(cudnnCreateTensorDescriptor(&resultTensorDesc2));
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
    checkCUDNN(
        cudnnSetActivationDescriptor(actiDesc, mode, CUDNN_PROPAGATE_NAN, 0.0));
    if (experts_num_layers == 1) {
      checkCUDNN(
          cudnnSetTensor4dDescriptor(resultTensorDesc1,
                                     CUDNN_TENSOR_NCHW,
                                     // CUDNN_DATA_FLOAT,
                                     cuda_to_cudnn_datatype(CUDA_R_32F),
                                     num_chosen_experts * effective_batch_size,
                                     out_dim,
                                     1,
                                     1));
    } else {
      checkCUDNN(
          cudnnSetTensor4dDescriptor(resultTensorDesc1,
                                     CUDNN_TENSOR_NCHW,
                                     // CUDNN_DATA_FLOAT,
                                     cuda_to_cudnn_datatype(CUDA_R_32F),
                                     num_chosen_experts * effective_batch_size,
                                     experts_internal_dim_size,
                                     1,
                                     1));
      checkCUDNN(
          cudnnSetTensor4dDescriptor(resultTensorDesc2,
                                     CUDNN_TENSOR_NCHW,
                                     // CUDNN_DATA_FLOAT,
                                     cuda_to_cudnn_datatype(CUDA_R_32F),
                                     num_chosen_experts * effective_batch_size,
                                     out_dim,
                                     1,
                                     1));
    }
  }
}
ExpertsMeta::~ExpertsMeta(void) {

  checkCUDA(cudaFree(sorted_indices));
  checkCUDA(cudaFree(original_indices));
  checkCUDA(cudaFree(non_zero_expert_labels));
  checkCUDA(cudaFree(temp_sequence));
  checkCUDA(cudaFree(exp_local_label_to_index));
  checkCUDA(cudaFree(expert_start_indexes));
  checkCUDA(cudaFree(num_assignments_per_expert));
  checkCUDA(cudaFree(destination_start_indices));
  checkCUDA(cudaFree(token_idx_array));
  checkCUDA(cudaFree(weight_idx_array1));
  checkCUDA(cudaFree(weight_idx_array2));
  checkCUDA(cudaFree(coefficient_idx_array));
  checkCUDA(cudaFree(output_idx_array));
  checkCUDA(cudaFree(dev_batch_outputs1));
  checkCUDA(cudaFree(dev_batch_outputs2));
  checkCUDA(cudaFree(bias_idx_array1));
  checkCUDA(cudaFree(bias_idx_array2));
  checkCUDA(cudaFree(batch_outputs1[0]));
  checkCUDA(cudaFree(batch_outputs2[0]));
  delete[] batch_outputs1;
  delete[] batch_outputs2;
  // Bias
  checkCUDA(cudaFree((void *)one_ptr));
  checkCUDA(cudaFree((void *)one_ptr_array));
  // Activation
  checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
  checkCUDNN(cudnnDestroyTensorDescriptor(resultTensorDesc1));
  checkCUDNN(cudnnDestroyTensorDescriptor(resultTensorDesc2));
}

}; // namespace FlexFlow

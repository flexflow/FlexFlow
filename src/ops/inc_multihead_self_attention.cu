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

#include "flexflow/ops/inc_multihead_self_attention.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;

void inference_kernel1(IncMultiHeadSelfAttentionMeta const *m,
                       BatchConfig const *bc,
                       float const *input_ptr,
                       float const *weight_ptr,
                       float *output_ptr,
                       cudaStream_t stream) {

  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  assert(m->qSize == m->vSize && m->qSize == m->kSize);
  cudaDataType_t data_type = ff_to_cuda_datatype(DT_FLOAT);
#if CUDA_VERSION >= 11000
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
  cudaDataType_t compute_type = CUDA_R_32F;
#endif
  // Compute (W^T)x matmul: einsum(ijkl,im->jmkl)
  // Weights: qSize x qProjSize x 3 x num_heads
  // Input: qSize x num_tokens
  // Output >>> qProjSize x num_tokens x 3 x num_heads
  int m_q = m->qProjSize;
  int m_k = m->kProjSize;
  int m_v = m->vProjSize;
  assert(m_q == m_k && m_k == m_v); // keep things simple for now
  int n = bc->num_active_tokens();
  int k = m->qSize;
  int lda = k, ldb = k, ldc_q = m_q, ldc_k = m_k, ldc_v = m_v;
  size_t strideA =
      m->weights_params; // need to also skip over all the parameters for each
                         // head, plus the unused W_o weights
  size_t strideB = 0;    // input stays the same for all heads.
  size_t strideC =
      (m_q + m_k + m_v) * n; // size of the output block for each head.
  // Q
  checkCUDA(cublasGemmStridedBatchedEx(m->handle.blas,
                                       CUBLAS_OP_T,
                                       CUBLAS_OP_N,
                                       m_q,
                                       n,
                                       k,
                                       &alpha,
                                       weight_ptr,
                                       data_type,
                                       lda,
                                       strideA,
                                       input_ptr,
                                       data_type,
                                       ldb,
                                       strideB,
                                       &beta,
                                       output_ptr,
                                       data_type,
                                       ldc_q,
                                       strideC,
                                       m->num_heads,
                                       compute_type,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  // K
  checkCUDA(cublasGemmStridedBatchedEx(m->handle.blas,
                                       CUBLAS_OP_T,
                                       CUBLAS_OP_N,
                                       m_k,
                                       n,
                                       k,
                                       &alpha,
                                       weight_ptr + m_q * k,
                                       data_type,
                                       lda,
                                       strideA,
                                       input_ptr,
                                       data_type,
                                       ldb,
                                       strideB,
                                       &beta,
                                       output_ptr + m_q * n,
                                       data_type,
                                       ldc_k,
                                       strideC,
                                       m->num_heads,
                                       compute_type,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  // V
  checkCUDA(cublasGemmStridedBatchedEx(m->handle.blas,
                                       CUBLAS_OP_T,
                                       CUBLAS_OP_N,
                                       m_v,
                                       n,
                                       k,
                                       &alpha,
                                       weight_ptr + (m_q + m_k) * k,
                                       data_type,
                                       lda,
                                       strideA,
                                       input_ptr,
                                       data_type,
                                       ldb,
                                       strideB,
                                       &beta,
                                       output_ptr + (m_q + m_k) * n,
                                       data_type,
                                       ldc_v,
                                       strideC,
                                       m->num_heads,
                                       compute_type,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

__global__ void store_kv_cache(float const *devQKVProjArray,
                               float *cache_ptr,
                               BatchConfig::token_idxs const *id_map,
                               int qProjSize,
                               int kProjSize,
                               int vProjSize,
                               int num_tokens,
                               int num_heads,
                               int max_seq_len,
                               bool k_cache) {
  CUDA_KERNEL_LOOP(i,
                   num_tokens * (k_cache ? kProjSize : vProjSize) * num_heads) {
    int proj_size = k_cache ? kProjSize : vProjSize;
    int head_idx = i / (num_tokens * proj_size);
    int token_idx = (i - head_idx * (num_tokens * proj_size)) / proj_size;
    int data_idx = i % proj_size;

    int qkv_block_size = (qProjSize + kProjSize + vProjSize) * num_tokens;
    int current_head_block_size =
        num_tokens * (k_cache ? qProjSize : qProjSize + kProjSize);
    float val =
        devQKVProjArray[head_idx * qkv_block_size + current_head_block_size +
                        token_idx * proj_size + data_idx];

    int const req_id = id_map[token_idx].request_index;
    int const tok_id = id_map[token_idx].token_position;

    cache_ptr[req_id * (num_heads * max_seq_len * proj_size) +
              head_idx * (max_seq_len * proj_size) + tok_id * proj_size +
              data_idx] = val;
  }
}

void inference_kernel2(IncMultiHeadSelfAttentionMeta const *m,
                       BatchConfig const *bc,
                       cudaStream_t stream) {
  int num_tokens = bc->num_active_tokens();
  if (num_tokens > 0) {
    int parallelism = m->kProjSize * num_tokens * m->num_heads;
    store_kv_cache<<<GET_BLOCKS(parallelism),
                     min(CUDA_NUM_THREADS, parallelism),
                     0,
                     stream>>>(m->devQKVProjArray,
                               m->keyCache,
                               m->dev_token2ids,
                               m->qProjSize,
                               m->kProjSize,
                               m->vProjSize,
                               num_tokens,
                               m->num_heads,
                               MAX_SEQ_LEN,
                               /* k_cache = */ true);
    parallelism = m->vProjSize * num_tokens * m->num_heads;
    store_kv_cache<<<GET_BLOCKS(parallelism),
                     min(CUDA_NUM_THREADS, parallelism),
                     0,
                     stream>>>(m->devQKVProjArray,
                               m->valueCache,
                               m->dev_token2ids,
                               m->qProjSize,
                               m->kProjSize,
                               m->vProjSize,
                               num_tokens,
                               m->num_heads,
                               MAX_SEQ_LEN,
                               /* k_cache = */ false);
  }
}

__global__ void fill_above_diagonal_square(float *matrix,
                                           int x_dim,
                                           int num_heads,
                                           int entries_above_diagonal,
                                           float value) {
  CUDA_KERNEL_LOOP(i, entries_above_diagonal * num_heads) {
    int head_idx = i / entries_above_diagonal;
    int y = (-1 + sqrt(8 * (float)i + 1)) / 2 + 1;
    int x = i - y * (y + 1) / 2;
    matrix[head_idx * x_dim * x_dim + x_dim * y + x] = value;
  }
}

__global__ void fill_last_entry_vector(float *matrix,
                                       int y_dim,
                                       int num_heads,
                                       float value) {
  // Fill last entry of each of the num_heads contiguous arrays of size y_dim
  CUDA_KERNEL_LOOP(i, num_heads) {
    matrix[i * y_dim + (y_dim - 1)] = value;
  }
}

void inference_kernel3(IncMultiHeadSelfAttentionMeta const *m,
                       BatchConfig const *bc,
                       float *output_ptr,
                       cudaStream_t stream) {
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  cudaDataType_t cublas_data_type = ff_to_cuda_datatype(DT_FLOAT);
#if CUDA_VERSION >= 11000
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
  cudaDataType_t compute_type = CUDA_R_32F;
#endif
  int num_requests = bc->num_active_requests();
  int num_tokens = bc->num_active_tokens();
  int tokens_previous_requests = 0;
  int tokens_prev_requests_squares = 0;
  int qkv_block_size =
      (m->qProjSize + m->kProjSize + m->vProjSize) * num_tokens;
  int kt_block_size = m->kProjSize * MAX_SEQ_LEN;
  int kt_req_block_size = kt_block_size * m->num_heads;
  int vt_block_size = m->vProjSize * MAX_SEQ_LEN;
  int vt_req_block_size = vt_block_size * m->num_heads;
  float alpha = 1.0f / (float)sqrt(m->kProjSize), beta = 0.0f;
  assert(m->qProjSize == m->kProjSize);

  for (int i = 0; i < num_requests; i++) {
    int num_new_tokens = bc->num_processing_tokens[i];
    int total_tokens = bc->token_last_available_idx[i] + 1;

    // Compute (QK^T/sqrt(d_k))
    int m_ = num_new_tokens;
    int n = total_tokens;
    int k = m->qProjSize;
    int lda = k, ldb = k, ldc = m_;
    int strideA = qkv_block_size;
    int strideB = kt_block_size;
    int strideC = num_new_tokens * total_tokens;
    // To get A, skip over Q entries from previous requests (same head)
    void const *A = (void const *)(m->devQKVProjArray +
                                   tokens_previous_requests * m->qProjSize);
    // To get B, skip over K entries from previous requests (all heads +
    // padding)
    void const *B = (void const *)(m->keyCache + i * kt_req_block_size);
    // To get C, skip over QK^T products from previous requests
    void *C =
        (void *)(m->qt_prods + m->num_heads * tokens_prev_requests_squares);

    checkCUDA(cublasGemmStridedBatchedEx(m->handle.blas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         m_,
                                         n,
                                         k,
                                         &alpha,
                                         A,
                                         cublas_data_type,
                                         lda,
                                         strideA,
                                         B,
                                         cublas_data_type,
                                         ldb,
                                         strideB,
                                         &beta,
                                         C,
                                         cublas_data_type,
                                         ldc,
                                         strideC,
                                         m->num_heads,
                                         compute_type,
                                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Fill all elements above diagonal in QT prods with -inf (FLT_MIN) to force
    // causal attention assume that number of rows (num new tokens) is either:
    // -> 1, in incremental phase, where each request has one more token
    // -> equal to number of columns (total number of tokens received so far) in
    // initialization phase
    assert(num_new_tokens == 1 || num_new_tokens == total_tokens);

    if (num_new_tokens == 1) {
      int parallelism = m->num_heads;
      fill_last_entry_vector<<<GET_BLOCKS(parallelism),
                               min(CUDA_NUM_THREADS, parallelism),
                               0,
                               stream>>>(
          (float *)C, total_tokens, m->num_heads, FLT_MIN);
    } else {
      int entries_above_diagonal = total_tokens * (total_tokens - 1) / 2;
      int parallelism = m->num_heads * entries_above_diagonal;
      fill_above_diagonal_square<<<GET_BLOCKS(parallelism),
                                   min(CUDA_NUM_THREADS, parallelism),
                                   0,
                                   stream>>>((float *)C,
                                             total_tokens,
                                             m->num_heads,
                                             entries_above_diagonal,
                                             FLT_MIN);
    }

    // Compute Softmax(QK^T/sqrt(d_k))
    cudnnTensorDescriptor_t qt_tensor;
    checkCUDNN(cudnnCreateTensorDescriptor(&qt_tensor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        qt_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, strideC, 1, 1, 1));
    alpha = 1.0f, beta = 0.0f;
    void *C_softmax = (void *)(m->qt_prods_softmax +
                               m->num_heads * tokens_prev_requests_squares);
    for (int attn_index = 0; attn_index < m->num_heads; attn_index++) {
      checkCUDNN(
          cudnnSoftmaxForward(m->handle.dnn,
                              CUDNN_SOFTMAX_ACCURATE,
                              CUDNN_SOFTMAX_MODE_CHANNEL,
                              &alpha,
                              qt_tensor,
                              (void *)((float *)C + attn_index * strideC),
                              &beta,
                              qt_tensor,
                              (void *)((float *)C + attn_index * strideC)));
    }

    // Matmul softmax(QK^T/sqrt(d_k)) by V
    alpha = 1.0f, beta = 0.0f;

    m_ = num_new_tokens;
    n = m->vProjSize;
    k = total_tokens;
    lda = m_, ldb = n, ldc = m_;
    strideA = num_new_tokens * total_tokens;
    strideB = vt_block_size;
    strideC = num_new_tokens * m->vProjSize;
    // To get A, skip over softmax(QK^T/sqrt(d_k)) entries from previous
    // requests (all heads)
    A = (void const *)((float *)C_softmax +
                       m->num_heads * tokens_prev_requests_squares);
    // To get B, skip over V^T entries from previous requests (all heads +
    // padding)
    B = (void const *)(m->valueCache + i * vt_req_block_size);
    // To get C, skip over softmax(QK^T/sqrt(d_k))V products from previous
    // requests
    C = (void *)(m->attn_heads +
                 tokens_previous_requests * m->num_heads * m->vProjSize);

    checkCUDA(cublasGemmStridedBatchedEx(m->handle.blas,
                                         CUBLAS_OP_N,
                                         CUBLAS_OP_T,
                                         m_,
                                         n,
                                         k,
                                         &alpha,
                                         A,
                                         cublas_data_type,
                                         lda,
                                         strideA,
                                         B,
                                         cublas_data_type,
                                         ldb,
                                         strideB,
                                         &beta,
                                         C,
                                         cublas_data_type,
                                         ldc,
                                         strideC,
                                         m->num_heads,
                                         compute_type,
                                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Project to output, save result directly on output tensor
    alpha = 1.0f, beta = 0.0f;

    m_ = num_new_tokens;
    n = m->oProjSize;
    k = m->vProjSize * m->num_heads;
    lda = m_, ldb = n, ldc = m_;
    A = (void const *)C;
    B = (void const *)m->W_out_contiguous;
    C = (void *)(output_ptr + tokens_previous_requests * m->oProjSize);

    checkCUDA(cublasGemmEx(m->handle.blas,
                           CUBLAS_OP_N,
                           CUBLAS_OP_T,
                           m_,
                           n,
                           k,
                           &alpha,
                           A,
                           cublas_data_type,
                           lda,
                           B,
                           cublas_data_type,
                           ldb,
                           &beta,
                           C,
                           cublas_data_type,
                           ldc,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    tokens_previous_requests += num_new_tokens;
    tokens_prev_requests_squares += num_new_tokens * total_tokens;
  }

  assert(tokens_previous_requests == num_tokens);
}

/*static*/
void IncMultiHeadSelfAttention::inference_kernel_wrapper(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    float const *input_ptr,
    float const *weight_ptr,
    float *output_ptr) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  // phase 1: Implement kernel to compute KQV for input tokens
  inference_kernel1(m, bc, input_ptr, weight_ptr, m->devQKVProjArray, stream);

  // phase 2: Update key/val cache
  cudaMemcpyAsync(m->dev_token2ids,
                  &(bc->token2ids.token_indexes),
                  bc->MAX_NUM_TOKENS * sizeof(BatchConfig::token_idxs),
                  cudaMemcpyHostToDevice,
                  stream);
  inference_kernel2(m, bc, stream);

  // phase 3: Compute attention score
  // 3 kernels for pahse 3: matmul1 - softmax - matmal2
  inference_kernel3(m, bc, output_ptr, stream);

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("IncMultiHeadSelfAttention forward time = %.2fms\n", elapsed);
    // print_tensor<3, float>(acc_query.ptr, acc_query.rect,
    // "[Attention:forward:query]"); print_tensor<3, float>(acc_output.ptr,
    // acc_output.rect, "[Attention:forward:output]");
  }
}

IncMultiHeadSelfAttentionMeta::IncMultiHeadSelfAttentionMeta(
    FFHandler handler,
    IncMultiHeadSelfAttention const *attn,
    float const *weight_ptr,
    Memory gpu_mem,
    int num_samples,
    int _num_heads)
    : OpMeta(handler, attn) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  // checkCUDNN(cudnnSetStream(handler.dnn, stream));

  qSize = attn->qSize;
  kSize = attn->kSize;
  vSize = attn->vSize;
  // assume dimensions match for now
  assert(qSize == kSize);
  assert(kSize == vSize);
  qProjSize = attn->qProjSize;
  kProjSize = attn->kProjSize;
  assert(qProjSize == kProjSize); // required for attention QK^T matmul
  vProjSize = attn->vProjSize;
  oProjSize = attn->oProjSize;
  num_heads = _num_heads;
  weights_params = (qSize * qProjSize + kSize * kProjSize + vSize * vProjSize +
                    oProjSize * (vProjSize > 0 ? vProjSize : vSize));
  weightSize = weights_params * num_heads * sizeof(float);

  // Currently do not support adding bias to key/value projection
  assert(!attn->add_bias_kv);

  // allocate memory for the seqArray and reserve space
  {
    size_t qkv_proj_dim = qProjSize + kProjSize + vProjSize;
    size_t qkv_max_proj_size =
        BatchConfig::MAX_NUM_TOKENS * qkv_proj_dim * num_heads;
    size_t key_cache_size =
        num_heads * kProjSize * BatchConfig::MAX_NUM_REQUESTS * MAX_SEQ_LEN;
    size_t value_cache_size =
        num_heads * vProjSize * BatchConfig::MAX_NUM_REQUESTS * MAX_SEQ_LEN;
    size_t token2ids_size = BatchConfig::MAX_NUM_TOKENS;
    size_t qt_prod_size =
        BatchConfig::MAX_NUM_TOKENS * BatchConfig::MAX_NUM_TOKENS * num_heads;
    size_t attn_heads_size =
        BatchConfig::MAX_NUM_TOKENS * num_heads * vProjSize;
    size_t W_out_block_size = oProjSize * (vProjSize > 0 ? vProjSize : vSize);
    size_t W_out_contiguous_size = W_out_block_size * num_heads;
    size_t totalSize =
        (qkv_max_proj_size + key_cache_size + value_cache_size +
         2 * qt_prod_size + attn_heads_size + W_out_contiguous_size) *
            sizeof(float) +
        token2ids_size *
            sizeof(BatchConfig::token_idxs); // more components will
                                             // be added here later

    Realm::Rect<1, coord_t> bounds(Realm::Point<1, coord_t>(0),
                                   Realm::Point<1, coord_t>(totalSize - 1));
    std::vector<size_t> field_sizes;
    field_sizes.push_back(sizeof(char));
    Realm::RegionInstance::create_instance(reserveInst,
                                           gpu_mem,
                                           bounds,
                                           field_sizes,
                                           0,
                                           Realm::ProfilingRequestSet())
        .wait();
    devQKVProjArray = (float *)reserveInst.pointer_untyped(0, sizeof(char));
    keyCache = (float *)devQKVProjArray + qkv_max_proj_size;
    valueCache = (float *)keyCache + key_cache_size;
    dev_token2ids = (BatchConfig::token_idxs *)(valueCache + value_cache_size);
    qt_prods = (float *)(dev_token2ids + token2ids_size);
    qt_prods_softmax = (float *)(qt_prods + qt_prod_size);
    attn_heads = (float *)qt_prods_softmax + qt_prod_size;
    W_out_contiguous = (float *)attn_heads + attn_heads_size;
    for (int h_idx = 0; h_idx < num_heads; h_idx++) {
      void *dest = (void *)(W_out_contiguous + W_out_block_size * h_idx);
      void const *src = (void const *)(weight_ptr + h_idx * weights_params +
                                       (qSize * qProjSize + kSize * kProjSize +
                                        vSize * vProjSize));
      checkCUDA(cudaMemcpy(dest,
                           src,
                           sizeof(float) * W_out_block_size,
                           cudaMemcpyDeviceToDevice));
    }
  }
}

IncMultiHeadSelfAttentionMeta::~IncMultiHeadSelfAttentionMeta(void) {
  reserveInst.destroy();
}

}; // namespace FlexFlow

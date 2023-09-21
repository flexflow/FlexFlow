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
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include "cuComplex.h"
#endif
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_kernels.h"
#include "flexflow/ops/tree_inc_multihead_self_attention.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;

using namespace Kernels::IncMultiHeadAttention;

namespace Kernels {
namespace TreeIncMultiHeadAttention {

template <typename DT>
__global__ void commit_tokens_kernel(
    DT const *devQKVProjArray,
    DT *kCache_ptr,
    DT *vCache_ptr,
    TreeVerifyBatchConfig::CommittedTokensInfo const *committedTokenInfos,
    int qProjSize,
    int kProjSize,
    int vProjSize,
    int num_tokens_to_commit,
    int num_active_tokens_in_last_batch,
    int num_q_heads,
    int num_kv_heads,
    int max_seq_len) {

  CUDA_KERNEL_LOOP(
      i, num_tokens_to_commit * (kProjSize + vProjSize) * num_kv_heads) {
    bool k_cache = i < (num_tokens_to_commit * kProjSize * num_kv_heads);
    int real_i =
        k_cache ? i : i - (num_tokens_to_commit * kProjSize * num_kv_heads);

    int proj_size = k_cache ? kProjSize : vProjSize;
    int data_idx = real_i % proj_size;
    int head_idx = real_i / (num_tokens_to_commit * proj_size);
    int token_pos =
        (real_i - head_idx * (num_tokens_to_commit * proj_size)) / proj_size;
    int token_idx_in_last_batch = committedTokenInfos[token_pos].token_index;
    assert(token_idx_in_last_batch < num_active_tokens_in_last_batch);

    int q_array_size =
        qProjSize * num_active_tokens_in_last_batch * num_q_heads;
    int k_array_size =
        kProjSize * num_active_tokens_in_last_batch * num_kv_heads;

    DT val =
        devQKVProjArray[q_array_size + (k_cache ? 0 : k_array_size) +
                        head_idx * proj_size * num_active_tokens_in_last_batch +
                        token_idx_in_last_batch * proj_size + data_idx];
    int const req_id = committedTokenInfos[token_pos].request_index;
    int const tok_id = committedTokenInfos[token_pos].token_depth;

    DT *cache_ptr = k_cache ? kCache_ptr : vCache_ptr;
    cache_ptr[req_id * (num_kv_heads * max_seq_len * proj_size) +
              head_idx * (max_seq_len * proj_size) + tok_id * proj_size +
              data_idx] = val;
  }
}

template <typename DT>
void commit_tokens(TreeIncMultiHeadSelfAttentionMeta const *m,
                   TreeVerifyBatchConfig const *bc,
                   cudaStream_t stream) {
  int num_tokens_to_commit = bc->num_tokens_to_commit;
  if (num_tokens_to_commit > 0) {
    int parallelism =
        (m->kProjSize + m->vProjSize) * num_tokens_to_commit * m->num_kv_heads;
    commit_tokens_kernel<<<GET_BLOCKS(parallelism),
                           min(CUDA_NUM_THREADS, parallelism),
                           0,
                           stream>>>(
        static_cast<DT *>(m->devQKVProjArray),
        static_cast<DT *>(m->keyCache),
        static_cast<DT *>(m->valueCache),
        m->committed_token_infos,
        m->qProjSize,
        m->kProjSize,
        m->vProjSize,
        num_tokens_to_commit,
        m->num_active_tokens, // number of active tokens in previous batch
        m->num_q_heads,
        m->num_kv_heads,
        BatchConfig::MAX_SEQ_LENGTH);
  }
}

template <typename DT>
__global__ void update_tree_branch_kv_cache(
    DT const *devQKVProjArray,
    DT *kCache_ptr,
    DT *vCache_ptr,
    TreeVerifyBatchConfig::PerTokenInfo const *tokenInfos,
    int qProjSize,
    int kProjSize,
    int vProjSize,
    int num_tokens_in_branch,
    int processed_tokens_in_batch,
    int total_tokens_in_batch,
    int num_q_heads,
    int num_kv_heads,
    int max_seq_len) {
  CUDA_KERNEL_LOOP(
      i, num_tokens_in_branch * (kProjSize + vProjSize) * num_kv_heads) {

    int q_array_size = qProjSize * total_tokens_in_batch * num_q_heads;
    int k_array_size = kProjSize * total_tokens_in_batch * num_kv_heads;

    bool k_cache = i < (num_tokens_in_branch * kProjSize * num_kv_heads);
    int real_i =
        k_cache ? i : i - (num_tokens_in_branch * kProjSize * num_kv_heads);

    int proj_size = k_cache ? kProjSize : vProjSize;
    int data_idx = real_i % proj_size;
    int token_idx =
        (real_i / proj_size) % num_tokens_in_branch; // index in the tree branch
    int head_idx = real_i / (proj_size * num_tokens_in_branch);

    token_idx += processed_tokens_in_batch; // get index in the whole batch
    DT val = devQKVProjArray[q_array_size + (k_cache ? 0 : k_array_size) +
                             head_idx * proj_size * total_tokens_in_batch +
                             token_idx * proj_size + data_idx];

    int const req_id = tokenInfos[token_idx].request_index;
    int const tok_id = tokenInfos[token_idx].abs_depth_in_request;
    DT *cache_ptr = k_cache ? kCache_ptr : vCache_ptr;

    cache_ptr[req_id * (num_kv_heads * max_seq_len * proj_size) +
              head_idx * (max_seq_len * proj_size) + tok_id * proj_size +
              data_idx] = val;
  }
}

template <typename DT>
__global__ void tree_fill_entries_above_diagonal(DT *matrix,
                                                 size_t new_tokens,
                                                 size_t total_tokens_in_request,
                                                 size_t num_q_heads,
                                                 DT value) {
  CUDA_KERNEL_LOOP(i, new_tokens * total_tokens_in_request * num_q_heads) {
    // size_t head_idx = i / (new_tokens * total_tokens_in_request);
    size_t src_idx = (i / new_tokens) % total_tokens_in_request;
    size_t dst_idx = i % new_tokens + total_tokens_in_request - new_tokens;
    // Casual Mask
    if (src_idx > dst_idx) {
      matrix[i] = value;
    }
  }
}

template <typename DT>
void compute_attention_kernel(TreeIncMultiHeadSelfAttentionMeta const *m,
                              TreeVerifyBatchConfig const *bc,
                              int shard_id,
                              DT *output_ptr,
                              DT const *bias_ptr,
                              DT const *weight_ptr,
                              cudaStream_t stream) {
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  cudaDataType_t cublas_data_type = ff_to_cuda_datatype(m->output_type[0]);
  cudnnDataType_t cudnn_data_type = ff_to_cudnn_datatype(m->output_type[0]);
  assert(data_type_size(m->output_type[0]) == sizeof(DT));
#if CUDA_VERSION >= 11000
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
  cudaDataType_t compute_type = cublas_data_type;
#endif
  // int num_requests = bc->num_active_requests();
  int processed_tokens_in_batch = 0;
  // int qkv_block_size =
  //     (m->qProjSize + m->kProjSize + m->vProjSize) * bc->num_active_tokens();
  int q_block_size = m->qProjSize * bc->num_active_tokens();
  int kt_block_size = m->kProjSize * BatchConfig::MAX_SEQ_LENGTH;
  int kt_req_block_size = kt_block_size * m->num_kv_heads;
  int vt_block_size = m->vProjSize * BatchConfig::MAX_SEQ_LENGTH;
  int vt_req_block_size = vt_block_size * m->num_kv_heads;
  assert(m->qProjSize == m->kProjSize);

  for (int i = 0; i < bc->MAX_NUM_REQUESTS; i++) {
    if (bc->request_completed[i]) {
      continue;
    }
    int last_token_idx_of_the_request =
        processed_tokens_in_batch + bc->requestsInfo[i].num_tokens_in_batch - 1;
    while (processed_tokens_in_batch <= last_token_idx_of_the_request) {
      int num_new_tokens = 1;
      int j = processed_tokens_in_batch;
      while ((j + 1 <= last_token_idx_of_the_request) &&
             (bc->tokensInfo[j].abs_depth_in_request + 1 ==
              bc->tokensInfo[j + 1].abs_depth_in_request)) {
        j++;
        num_new_tokens++;
      }

      int total_tokens_in_request = bc->tokensInfo[j].abs_depth_in_request + 1;
      assert(num_new_tokens >= 1 && total_tokens_in_request >= num_new_tokens);
      {
        // update K-V cache
        int parallelism =
            (m->kProjSize + m->vProjSize) * num_new_tokens * m->num_kv_heads;
        update_tree_branch_kv_cache<<<GET_BLOCKS(parallelism),
                                      min(CUDA_NUM_THREADS, parallelism),
                                      0,
                                      stream>>>(
            static_cast<DT *>(m->devQKVProjArray),
            static_cast<DT *>(m->keyCache),
            static_cast<DT *>(m->valueCache),
            m->token_infos,
            m->qProjSize,
            m->kProjSize,
            m->vProjSize,
            num_new_tokens,            // num_tokens_in_branch
            processed_tokens_in_batch, // num_processed_tokens_in_batch
            m->num_active_tokens,      // total_tokens_in_batch
            m->num_q_heads,
            m->num_kv_heads,
            BatchConfig::MAX_SEQ_LENGTH);
      }

      // bc->token_last_available_idx[i] + 1;
      // Compute (QK^T/sqrt(d_k))
      int m_ = num_new_tokens;
      int n = total_tokens_in_request;
      int k = m->qProjSize;
      int lda = k, ldb = k, ldc = m_;
      int strideA = q_block_size;
      int strideB = kt_block_size;
      int strideC = num_new_tokens * total_tokens_in_request;

      // a flag of using this scaling alpha
      DT alpha = 1.0f, beta = 0.0f;
      if (*m->qk_prod_scaling) {
        alpha = static_cast<DT>(1.0f / sqrt(m->kProjSize));
      }
      // To get A, skip over Q entries from previous requests (same head)
      DT const *A = static_cast<DT *>(m->devQKVProjArray) +
                    processed_tokens_in_batch * m->qProjSize;
      // To get B, skip over K entries from previous requests (all heads +
      // padding)
      DT const *B = static_cast<DT *>(m->keyCache) + i * kt_req_block_size;
      // To get C, skip over QK^T products from previous requests
      DT *C = static_cast<DT *>(m->qk_prods);

      if (m->num_q_heads == m->num_kv_heads) {
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
                                             m->num_q_heads,
                                             compute_type,
                                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      } else {
        strideB = 0;
        int one_step_heads = m->num_q_heads / m->num_kv_heads;
        for (int step = 0; step < m->num_kv_heads; step++) {
          checkCUDA(
              cublasGemmStridedBatchedEx(m->handle.blas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         m_,
                                         n,
                                         k,
                                         &alpha,
                                         A + step * strideA * one_step_heads,
                                         cublas_data_type,
                                         lda,
                                         strideA,
                                         B + step * kt_block_size,
                                         cublas_data_type,
                                         ldb,
                                         strideB,
                                         &beta,
                                         C + step * strideC * one_step_heads,
                                         cublas_data_type,
                                         ldc,
                                         strideC,
                                         one_step_heads,
                                         compute_type,
                                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
      }
      // add alibi position bias to qk production
      // add alibi position bias to qk production
      if (*m->position_bias) {
        size_t parallelism =
            m->num_q_heads * total_tokens_in_request * num_new_tokens;
        apply_position_bias_qkprd<<<GET_BLOCKS(parallelism),
                                    min((size_t)CUDA_NUM_THREADS, parallelism),
                                    0,
                                    stream>>>(C,
                                              num_new_tokens,
                                              total_tokens_in_request,
                                              m->num_q_heads,
                                              m->global_num_q_heads,
                                              shard_id);
      }

      // Fill all elements above diagonal in qk prods with -inf to force
      // causal attention.
      assert(num_new_tokens <= total_tokens_in_request);
      if (num_new_tokens > 1) {
        size_t parallelism =
            m->num_q_heads * num_new_tokens * total_tokens_in_request;
        tree_fill_entries_above_diagonal<<<GET_BLOCKS(parallelism),
                                           min((size_t)CUDA_NUM_THREADS,
                                               parallelism),
                                           0,
                                           stream>>>(
            C,
            num_new_tokens,
            total_tokens_in_request,
            m->num_q_heads,
            static_cast<DT>(-INFINITY));
      }
      // Compute Softmax(QK^T/sqrt(d_k))
      // Before modifying the parameters below, make sure to read the following
      // description of the CUDNN_TENSOR_NCHW tensor layout, from
      // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnTensorFormat_t:
      // This tensor format specifies that the data is laid out in the following
      // order: batch size, feature maps, rows, columns. The strides are
      // implicitly defined in such a way that the data are contiguous in memory
      // with no padding between images, feature maps, rows, and columns; the
      // columns are the inner dimension and the images are the outermost
      // dimension.
      int n_param = m->num_q_heads;
      int c_param = total_tokens_in_request;
      int h_param = 1;
      int w_param = num_new_tokens;
      checkCUDNN(cudnnSetTensor4dDescriptor(m->qk_tensor,
                                            CUDNN_TENSOR_NCHW,
                                            cudnn_data_type,
                                            n_param,
                                            c_param,
                                            h_param,
                                            w_param));
      float softmax_alpha = 1.0f, softmax_beta = 0.0f;
      DT *C_softmax = static_cast<DT *>(m->qk_prods_softmax);
      // The softmax operation below is executed according to the
      // CUDNN_SOFTMAX_MODE_CHANNEL, which is also described in the docs: The
      // softmax operation is computed per spatial location (H,W) per image (N)
      // across dimension C.
      checkCUDNN(cudnnSoftmaxForward(m->handle.dnn,
                                     CUDNN_SOFTMAX_ACCURATE,
                                     CUDNN_SOFTMAX_MODE_CHANNEL,
                                     &softmax_alpha,
                                     m->qk_tensor,
                                     C,
                                     &softmax_beta,
                                     m->qk_tensor,
                                     C_softmax));
      // Matmul softmax(QK^T/sqrt(d_k)) by V
      alpha = 1.0f, beta = 0.0f;
      m_ = num_new_tokens;
      n = m->vProjSize;
      k = total_tokens_in_request;
      lda = m_, ldb = n, ldc = m_;
      strideA = num_new_tokens * total_tokens_in_request;
      strideB = vt_block_size;
      strideC = num_new_tokens * m->vProjSize;
      // To get A, skip over softmax(QK^T/sqrt(d_k)) entries from previous
      // requests (all heads)
      A = C_softmax;
      // To get B, skip over V^T entries from previous requests (all heads +
      // padding)
      B = static_cast<DT *>(m->valueCache) + i * vt_req_block_size;
      // To get C, skip over softmax(QK^T/sqrt(d_k))V products from previous
      // requests
      C = static_cast<DT *>(m->attn_heads) +
          processed_tokens_in_batch * m->num_q_heads * m->vProjSize;

      if (m->num_q_heads == m->num_kv_heads) {
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
                                             m->num_q_heads,
                                             compute_type,
                                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      } else {
        int one_step_heads = m->num_q_heads / m->num_kv_heads;
        strideB = 0;
        for (int step = 0; step < m->num_kv_heads; step++) {
          checkCUDA(
              cublasGemmStridedBatchedEx(m->handle.blas,
                                         CUBLAS_OP_N,
                                         CUBLAS_OP_T,
                                         m_,
                                         n,
                                         k,
                                         &alpha,
                                         A + step * one_step_heads * strideA,
                                         cublas_data_type,
                                         lda,
                                         strideA,
                                         B + step * vt_block_size,
                                         cublas_data_type,
                                         ldb,
                                         strideB,
                                         &beta,
                                         C + step * one_step_heads * strideC,
                                         cublas_data_type,
                                         ldc,
                                         strideC,
                                         one_step_heads,
                                         compute_type,
                                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
      }

      // Project to output, save result directly on output tensor
      alpha = 1.0f, beta = 0.0f;
      m_ = m->oProjSize;
      k = m->vProjSize * m->num_q_heads;
      n = num_new_tokens;
      lda = k, ldb = n, ldc = m_;
      A = weight_ptr + m->qSize * (m->qProjSize * m->num_q_heads +
                                   m->kProjSize * m->num_kv_heads +
                                   m->vProjSize * m->num_kv_heads);
      B = C;
      C = static_cast<DT *>(output_ptr) +
          processed_tokens_in_batch * m->oProjSize;

      checkCUDA(cublasGemmEx(m->handle.blas,
                             CUBLAS_OP_T,
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
      processed_tokens_in_batch += num_new_tokens;
    }
    // Before moving to the next request
    // check that we have finished all tokens of the request
    assert(last_token_idx_of_the_request + 1 == processed_tokens_in_batch);
  }
  if (*m->final_bias && shard_id == 0) {
    int parallelism = m->oProjSize * processed_tokens_in_batch;
    int qkv_weight_size = m->qProjSize * m->global_num_q_heads +
                          m->kProjSize * m->global_num_kv_heads +
                          m->vProjSize * m->global_num_kv_heads;
    apply_proj_bias_w<<<GET_BLOCKS(parallelism),
                        min(CUDA_NUM_THREADS, parallelism),
                        0,
                        stream>>>(output_ptr,
                                  bias_ptr,
                                  processed_tokens_in_batch,
                                  qkv_weight_size,
                                  m->oProjSize);
  }

  assert(processed_tokens_in_batch == bc->num_active_tokens());
}

template <typename DT>
void inference_kernel(TreeIncMultiHeadSelfAttentionMeta *m,
                      TreeVerifyBatchConfig const *bc,
                      int shard_id,
                      DT const *input_ptr,
                      DT const *weight_ptr,
                      DT *output_ptr,
                      DT const *bias_ptr,
                      cudaStream_t stream) {
  // additional processing for weight uploading
  if (m->handle.offload_reserve_space != nullptr) {
    // Note that we update weight_ptr and bias_ptr when uploading weight and
    // bias
    cudaMemcpyAsync(m->weight_ptr,
                    weight_ptr,
                    m->weightSize,
                    cudaMemcpyHostToDevice,
                    stream);
    weight_ptr = static_cast<DT *>(m->weight_ptr);
    if (m->biasSize > 0) {
      cudaMemcpyAsync(
          m->bias_ptr, bias_ptr, m->biasSize, cudaMemcpyHostToDevice, stream);
      bias_ptr = static_cast<DT *>(m->bias_ptr);
    }
  }
  // copy committed tokens info to GPU for the commit_tokens kernel
  // Note that m->num_active_tokens stores the number of active
  // tokens in the previous batch, which is needed for committing
  // keys/values to the key-value cache
  cudaMemcpyAsync(m->committed_token_infos,
                  &(bc->committed_tokens),
                  bc->num_tokens_to_commit *
                      sizeof(TreeVerifyBatchConfig::CommittedTokensInfo),
                  cudaMemcpyHostToDevice,
                  stream);
  commit_tokens<DT>(m, bc, stream);

  // After commit we update m->num_active_tokens to be the number of active
  // tokens for the current batch
  m->num_active_tokens = bc->num_active_tokens();

  // here because we need postion info in infernece 1
  if (m->offload && m->biasSize > 0) {
    cudaMemcpyAsync(
        m->bias_ptr, bias_ptr, m->biasSize, cudaMemcpyHostToDevice, stream);
    bias_ptr = static_cast<DT *>(m->bias_ptr);
  }
  cudaMemcpyAsync(m->token_infos,
                  &(bc->tokensInfo),
                  bc->MAX_NUM_TOKENS *
                      sizeof(TreeVerifyBatchConfig::PerTokenInfo),
                  cudaMemcpyHostToDevice,
                  stream);
  // phase 1: Implement kernel to compute KQV for input tokens
  compute_qkv_kernel(m,
                     bc,
                     shard_id,
                     input_ptr,
                     weight_ptr,
                     static_cast<DT *>(m->devQKVProjArray),
                     bias_ptr,
                     stream);

  // phase 2: No need to update key/val cache
  // IncMultiHeadSelfAttention::update_kv_cache_kernel(
  //    m, bc, stream);

  // phase 3: Compute attention score
  // 3 kernels for pahse 3: matmul1 - softmax - matmal2
  compute_attention_kernel(
      m, bc, shard_id, output_ptr, bias_ptr, weight_ptr, stream);
}

} // namespace TreeIncMultiHeadAttention
} // namespace Kernels

/*static*/
void TreeIncMultiHeadSelfAttention::inference_kernel_wrapper(
    TreeIncMultiHeadSelfAttentionMeta *m,
    TreeVerifyBatchConfig const *bc,
    int shard_id,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorR const &weight,
    GenericTensorAccessorW const &output,
    GenericTensorAccessorR const &bias) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  bool use_bias = *m->qkv_bias || *m->final_bias;

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  // assert(input.data_type == weight.data_type);
  assert(input.data_type == output.data_type);
  if (use_bias) {
    assert(input.data_type == bias.data_type);
  }

  if (input.data_type == DT_HALF) {
    if (m->offload) {
      pre_build_weight_kernel<half>(m, weight, input.data_type, stream);
    }

    half const *bias_ptr =
        use_bias ? bias.get_half_ptr() : static_cast<half const *>(nullptr);
    Kernels::TreeIncMultiHeadAttention::inference_kernel(
        m,
        bc,
        shard_id,
        input.get_half_ptr(),
        m->offload ? static_cast<half *>(m->weight_ptr) : weight.get_half_ptr(),
        output.get_half_ptr(),
        bias_ptr,
        stream);
  } else if (input.data_type == DT_FLOAT) {
    if (m->offload) {
      pre_build_weight_kernel<float>(m, weight, input.data_type, stream);
    }
    float const *bias_ptr =
        use_bias ? bias.get_float_ptr() : static_cast<float const *>(nullptr);
    Kernels::TreeIncMultiHeadAttention::inference_kernel(
        m,
        bc,
        shard_id,
        input.get_float_ptr(),
        m->offload ? static_cast<float *>(m->weight_ptr)
                   : weight.get_float_ptr(),
        output.get_float_ptr(),
        bias_ptr,
        stream);
  } else {
    assert(false && "Unspported data type");
  }

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("TreeIncMultiHeadSelfAttention forward time = %.2fms\n", elapsed);
    // print_tensor<3, float>(acc_query.ptr, acc_query.rect,
    // "[Attention:forward:query]"); print_tensor<3, float>(acc_output.ptr,
    // acc_output.rect, "[Attention:forward:output]");
  }
}

TreeIncMultiHeadSelfAttentionMeta::TreeIncMultiHeadSelfAttentionMeta(
    FFHandler handler,
    TreeIncMultiHeadSelfAttention const *attn,
    GenericTensorAccessorR const &weight,
    MemoryAllocator &gpu_mem_allocator,
    int num_samples,
    int _num_q_heads,
    int _num_kv_heads)
    : IncMultiHeadSelfAttentionMeta(handler,
                                    TREE_VERIFY_MODE,
                                    attn,
                                    attn->qSize,
                                    attn->kSize,
                                    attn->vSize,
                                    attn->qProjSize,
                                    attn->kProjSize,
                                    attn->vProjSize,
                                    attn->oProjSize,
                                    attn->apply_rotary_embedding,
                                    attn->qkv_bias,
                                    attn->scaling_query,
                                    attn->qk_prod_scaling,
                                    attn->position_bias,
                                    attn->final_bias,
                                    attn->scaling_factor,
                                    weight,
                                    gpu_mem_allocator,
                                    num_samples,
                                    attn->num_q_heads,
                                    attn->num_kv_heads,
                                    _num_q_heads,
                                    _num_kv_heads,
                                    attn->quantization_type,
                                    attn->offload),
      num_active_tokens(0) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDNN(cudnnSetStream(handler.dnn, stream));

  // allocate memory for the seqArray and reserve space
  {
    size_t committed_tokeninfo_size = TreeVerifyBatchConfig::MAX_NUM_TOKENS;
    size_t total_size = committed_tokeninfo_size *
                        sizeof(TreeVerifyBatchConfig::CommittedTokensInfo);
    if (offload) {
      // assert that we have enough reserved work space left
      assert(gpu_mem_allocator.reserved_total_size -
                 gpu_mem_allocator.reserved_allocated_size >=
             total_size);
      committed_token_infos =
          gpu_mem_allocator
              .allocate_reserved<TreeVerifyBatchConfig::CommittedTokensInfo>(
                  committed_tokeninfo_size);
    } else {
      gpu_mem_allocator.create_legion_instance(committed_token_reserve_inst,
                                               total_size);
      committed_token_infos =
          gpu_mem_allocator
              .allocate_instance<TreeVerifyBatchConfig::CommittedTokensInfo>(
                  committed_tokeninfo_size);
    }
  }

  cudaStreamSynchronize(stream);
}

TreeIncMultiHeadSelfAttentionMeta::~TreeIncMultiHeadSelfAttentionMeta(void) {
  if (committed_token_reserve_inst != Realm::RegionInstance::NO_INST) {
    committed_token_reserve_inst.destroy();
  }
}

}; // namespace FlexFlow

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

#include "flexflow/ops/tree_inc_multihead_self_attention.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_kernels.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

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
    int max_seq_len,
    int hidden_size) {

  CUDA_KERNEL_LOOP(i, num_tokens_to_commit * hidden_size * 2) {

    int token_pos = i / (hidden_size * KV_WEIGHT_NUM);
    int token_idx_in_last_batch = committedTokenInfos[token_pos].token_index;
    int offset = i % hidden_size;
    assert(token_idx_in_last_batch < num_active_tokens_in_last_batch);

    size_t val_idx =
        token_idx_in_last_batch * 3 * hidden_size + hidden_size + offset;

    DT kVal = devQKVProjArray[val_idx];
    DT vVal = devQKVProjArray[val_idx + hidden_size];

    int const req_id = committedTokenInfos[token_pos].request_index;
    int const tok_id = committedTokenInfos[token_pos].token_depth;

    kCache_ptr[req_id * (hidden_size * max_seq_len) + tok_id * hidden_size +
               offset] = kVal;
    vCache_ptr[req_id * (hidden_size * max_seq_len) + tok_id * hidden_size +
               offset] = vVal;
  }
}

template <typename DT>
void commit_tokens(TreeIncMultiHeadSelfAttentionMeta const *m,
                   TreeVerifyBatchConfig const *bc,
                   hipStream_t stream) {
  int num_tokens_to_commit = bc->num_tokens_to_commit;
  if (num_tokens_to_commit > 0) {
    int parallelism = m->hidden_size * KV_WEIGHT_NUM * num_tokens_to_commit;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(commit_tokens_kernel<DT>),
        GET_BLOCKS(parallelism),
        min(CUDA_NUM_THREADS, parallelism),
        0,
        stream,
        static_cast<DT *>(m->devQKVProjArray),
        static_cast<DT *>(m->keyCache),
        static_cast<DT *>(m->valueCache),
        m->committed_token_infos,
        m->qProjSize,
        m->kProjSize,
        m->vProjSize,
        num_tokens_to_commit,
        m->num_active_tokens, // number of active tokens in previous batch
        BatchConfig::max_sequence_length(),
        m->hidden_size);
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
    int max_seq_len,
    int hidden_size) {
  CUDA_KERNEL_LOOP(i, num_tokens_in_branch * hidden_size * 2) {
    int token_idx = i / (hidden_size * KV_WEIGHT_NUM);
    int offset = i % hidden_size;

    token_idx += processed_tokens_in_batch; // get index in the whole batch
    size_t val_idx = token_idx * 3 * hidden_size + hidden_size + offset;
    DT kVal = devQKVProjArray[val_idx];
    DT vVal = devQKVProjArray[val_idx + hidden_size];

    int const req_id = tokenInfos[token_idx].request_index;
    int const tok_id = tokenInfos[token_idx].abs_depth_in_request;
    kCache_ptr[req_id * (hidden_size * max_seq_len) + tok_id * hidden_size +
               offset] = kVal;
    vCache_ptr[req_id * (hidden_size * max_seq_len) + tok_id * hidden_size +
               offset] = vVal;
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
                              hipStream_t stream) {
  checkCUDA(hipblasSetStream(m->handle.blas, stream));
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));
  hipblasDatatype_t hipblas_data_type = ff_to_cuda_datatype(m->output_type[0]);
  miopenDataType_t miopen_data_type = ff_to_cudnn_datatype(m->output_type[0]);
  assert(data_type_size(m->output_type[0]) == sizeof(DT));
#if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
  hipblasDatatype_t compute_type = hipblas_data_type;
#else
  // TODO: currently use the hipblas_data_type
  // cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
  hipblasDatatype_t compute_type = hipblas_data_type;
#endif
  // int num_requests = bc->num_active_requests();
  int processed_tokens_in_batch = 0;
  // int qkv_block_size =
  //     (m->qProjSize + m->kProjSize + m->vProjSize) * bc->num_active_tokens();
  int q_block_size = m->qProjSize;
  int kt_block_size = m->kProjSize;
  int kt_req_block_size =
      kt_block_size * m->num_q_heads * BatchConfig::max_sequence_length();
  int vt_block_size = m->vProjSize;
  int vt_req_block_size =
      vt_block_size * m->num_q_heads * BatchConfig::max_sequence_length();
  assert(m->qProjSize == m->kProjSize);

  for (int i = 0; i < bc->max_requests_per_batch(); i++) {
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
        int parallelism = m->hidden_size * KV_WEIGHT_NUM * num_new_tokens;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(update_tree_branch_kv_cache<DT>),
            GET_BLOCKS(parallelism),
            min(CUDA_NUM_THREADS, parallelism),
            0,
            stream,
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
            BatchConfig::max_sequence_length(),
            m->hidden_size);
      }

      // bc->token_last_available_idx[i] + 1;
      // Compute (QK^T/sqrt(d_k))
      int m_ = num_new_tokens;
      int n = total_tokens_in_request;
      int k = m->qProjSize;
      int lda = k * m->num_q_heads * QKV_WEIGHT_NUM, ldb = k * m->num_q_heads,
          ldc = m_;
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
                    processed_tokens_in_batch * m->qProjSize * m->num_q_heads *
                        QKV_WEIGHT_NUM;
      // To get B, skip over K entries from previous requests (all heads +
      // padding)
      DT const *B = static_cast<DT *>(m->keyCache) + i * kt_req_block_size;
      // To get C, skip over QK^T products from previous requests
      DT *C = static_cast<DT *>(m->qk_prods);

      checkCUDA(hipblasGemmStridedBatchedEx(m->handle.blas,
                                            HIPBLAS_OP_T,
                                            HIPBLAS_OP_N,
                                            m_,
                                            n,
                                            k,
                                            &alpha,
                                            A,
                                            hipblas_data_type,
                                            lda,
                                            strideA,
                                            B,
                                            hipblas_data_type,
                                            ldb,
                                            strideB,
                                            &beta,
                                            C,
                                            hipblas_data_type,
                                            ldc,
                                            strideC,
                                            m->num_q_heads,
                                            compute_type,
                                            HIPBLAS_GEMM_DEFAULT));

      if (*m->position_bias) {
        size_t parallelism =
            m->num_q_heads * total_tokens_in_request * num_new_tokens;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(apply_position_bias_qkprd<DT>),
                           GET_BLOCKS(parallelism),
                           min((size_t)CUDA_NUM_THREADS, parallelism),
                           0,
                           stream,
                           C,
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
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(tree_fill_entries_above_diagonal<DT>),
            GET_BLOCKS(parallelism),
            min((size_t)CUDA_NUM_THREADS, parallelism),
            0,
            stream,
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
      checkCUDNN(miopenSet4dTensorDescriptor(
          m->qk_tensor, miopen_data_type, n_param, c_param, h_param, w_param));
      float softmax_alpha = 1.0f, softmax_beta = 0.0f;
      DT *C_softmax = static_cast<DT *>(m->qk_prods_softmax);
      // The softmax operation below is executed according to the
      // CUDNN_SOFTMAX_MODE_CHANNEL, which is also described in the docs: The
      // softmax operation is computed per spatial location (H,W) per image (N)
      // across dimension C.
      checkCUDNN(miopenSoftmaxForward_V2(m->handle.dnn,
                                         &softmax_alpha,
                                         m->qk_tensor,
                                         C,
                                         &softmax_beta,
                                         m->qk_tensor,
                                         C_softmax,
                                         MIOPEN_SOFTMAX_ACCURATE,
                                         MIOPEN_SOFTMAX_MODE_CHANNEL));
      // Matmul softmax(QK^T/sqrt(d_k)) by V
      alpha = 1.0f, beta = 0.0f;
      m_ = num_new_tokens;
      n = m->vProjSize;
      k = total_tokens_in_request;
      lda = m_, ldb = n * m->num_q_heads, ldc = m_;
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

      checkCUDA(hipblasGemmStridedBatchedEx(m->handle.blas,
                                            HIPBLAS_OP_N,
                                            HIPBLAS_OP_T,
                                            m_,
                                            n,
                                            k,
                                            &alpha,
                                            A,
                                            hipblas_data_type,
                                            lda,
                                            strideA,
                                            B,
                                            hipblas_data_type,
                                            ldb,
                                            strideB,
                                            &beta,
                                            C,
                                            hipblas_data_type,
                                            ldc,
                                            strideC,
                                            m->num_q_heads,
                                            compute_type,
                                            HIPBLAS_GEMM_DEFAULT));

      // Project to output, save result directly on output tensor
      alpha = 1.0f, beta = 0.0f;
      m_ = m->oProjSize;
      k = m->vProjSize * m->num_q_heads;
      n = num_new_tokens;
      lda = k, ldb = n, ldc = m_;
      A = weight_ptr + m->qSize * (m->qProjSize * m->num_q_heads +
                                   m->kProjSize * m->num_q_heads +
                                   m->vProjSize * m->num_q_heads);
      B = C;
      C = static_cast<DT *>(output_ptr) +
          processed_tokens_in_batch * m->oProjSize;

      checkCUDA(hipblasGemmEx(m->handle.blas,
                              HIPBLAS_OP_T,
                              HIPBLAS_OP_T,
                              m_,
                              n,
                              k,
                              &alpha,
                              A,
                              hipblas_data_type,
                              lda,
                              B,
                              hipblas_data_type,
                              ldb,
                              &beta,
                              C,
                              hipblas_data_type,
                              ldc,
                              compute_type,
                              HIPBLAS_GEMM_DEFAULT));
      processed_tokens_in_batch += num_new_tokens;
    }
    // Before moving to the next request
    // check that we have finished all tokens of the request
    assert(last_token_idx_of_the_request + 1 == processed_tokens_in_batch);
  }
  if (*m->final_bias && shard_id == 0) {
    int parallelism = m->oProjSize * processed_tokens_in_batch;
    int qkv_weight_size = m->qProjSize * m->global_num_q_heads +
                          m->kProjSize * m->global_num_q_heads +
                          m->vProjSize * m->global_num_q_heads;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(apply_proj_bias_w<DT>),
                       GET_BLOCKS(parallelism),
                       min(CUDA_NUM_THREADS, parallelism),
                       0,
                       stream,
                       output_ptr,
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
                      hipStream_t stream) {
  // additional processing for weight uploading
  if (m->handle.offload_reserve_space != nullptr) {
    // Note that we update weight_ptr and bias_ptr when uploading weight and
    // bias
    checkCUDA(hipMemcpyAsync(m->weight_ptr,
                             weight_ptr,
                             m->weightSize,
                             hipMemcpyHostToDevice,
                             stream));
    weight_ptr = static_cast<DT *>(m->weight_ptr);
    if (m->biasSize > 0) {
      checkCUDA(hipMemcpyAsync(
          m->bias_ptr, bias_ptr, m->biasSize, hipMemcpyHostToDevice, stream));
      bias_ptr = static_cast<DT *>(m->bias_ptr);
    }
  }
  // copy committed tokens info to GPU for the commit_tokens kernel
  // Note that m->num_active_tokens stores the number of active
  // tokens in the previous batch, which is needed for committing
  // keys/values to the key-value cache
  checkCUDA(
      hipMemcpyAsync(m->committed_token_infos,
                     &(bc->committed_tokens),
                     bc->num_tokens_to_commit *
                         sizeof(TreeVerifyBatchConfig::CommittedTokensInfo),
                     hipMemcpyHostToDevice,
                     stream));
  commit_tokens<DT>(m, bc, stream);

  // After commit we update m->num_active_tokens to be the number of active
  // tokens for the current batch
  m->num_active_tokens = bc->num_active_tokens();

  // here because we need postion info in infernece 1
  if (m->offload && m->biasSize > 0) {
    checkCUDA(hipMemcpyAsync(
        m->bias_ptr, bias_ptr, m->biasSize, hipMemcpyHostToDevice, stream));
    bias_ptr = static_cast<DT *>(m->bias_ptr);
  }
  checkCUDA(hipMemcpyAsync(m->token_infos,
                           &(bc->tokensInfo),
                           bc->num_active_tokens() *
                               sizeof(TreeVerifyBatchConfig::PerTokenInfo),
                           hipMemcpyHostToDevice,
                           stream));
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
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  bool use_bias = *m->qkv_bias || *m->final_bias;

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
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
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
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
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDNN(miopenSetStream(handler.dnn, stream));

  // allocate memory for the seqArray and reserve space
  {
    int max_tokens_per_batch = BatchConfig::max_tokens_per_batch();
    size_t committed_tokeninfo_size = max_tokens_per_batch;
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

  checkCUDA(hipStreamSynchronize(stream));
}

TreeIncMultiHeadSelfAttentionMeta::~TreeIncMultiHeadSelfAttentionMeta(void) {
  if (committed_token_reserve_inst != Realm::RegionInstance::NO_INST) {
    committed_token_reserve_inst.destroy();
  }
}

}; // namespace FlexFlow

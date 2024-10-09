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

#include "flexflow/ops/spec_inc_multihead_self_attention.h"
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
namespace SpecIncMultiHeadSelfAttention {

template <typename DT>
__global__ void spec_inc_store_kv_cache(
    DT const *devQKVProjArray,
    DT *kCache_ptr,
    DT *vCache_ptr,
    BatchConfig::PerTokenInfo *tokenInfos,
    BatchConfig::PerRequestInfo *requestInfo,
    BeamSearchBatchConfig::BeamSearchPerTokenInfo *beamTokenInfos,
    BeamSearchBatchConfig::BeamSearchPerRequestInfo *beamRequestInfos,
    BatchConfig::BitMask *causalMask,
    int qProjSize,
    int kProjSize,
    int vProjSize,
    int num_tokens,
    int max_seq_len,
    bool is_root,
    int hidden_size) {
  CUDA_KERNEL_LOOP(i, num_tokens * hidden_size) {
    int token_idx = i / (hidden_size);
    int offset = i % hidden_size;

    size_t val_idx =
        token_idx * QKV_WEIGHT_NUM * hidden_size + hidden_size + offset;

    DT kVal = devQKVProjArray[val_idx];
    DT vVal = devQKVProjArray[val_idx + hidden_size];

    int const req_id = tokenInfos[token_idx].request_index;
    // int const tok_id = tokenInfos[token_idx].abs_depth_in_request;

    int const request_token_offset =
        requestInfo[req_id].first_token_offset_in_batch;

    BatchConfig::BitMask bitmask = causalMask[req_id];

    // if prompt token -> token id
    // if tree token:

    int const cache_idx = bitmask.prompt_size + bitmask.non_tree_cache_size +
                          bitmask.tree_size - 1 - bitmask.this_layer_size +
                          token_idx - request_token_offset;

    kCache_ptr[req_id * (hidden_size * max_seq_len) + (cache_idx)*hidden_size +
               offset] = kVal;
    vCache_ptr[req_id * (hidden_size * max_seq_len) + (cache_idx)*hidden_size +
               offset] = vVal;
  }
}

template <typename DT>
void update_kv_cache_kernel(SpecIncMultiHeadSelfAttentionMeta const *m,
                            BeamSearchBatchConfig const *bc,
                            hipStream_t stream) {
  int num_tokens = bc->num_active_infr_tokens();
  int curr_depth = bc->beamRequestsInfo[0].current_depth;
  if (num_tokens > 0) {
    int parallelism = m->hidden_size * KV_WEIGHT_NUM * num_tokens;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(spec_store_kv_cache<DT>),
                       GET_BLOCKS(parallelism),
                       min(CUDA_NUM_THREADS, parallelism),
                       0,
                       stream,
                       static_cast<DT *>(m->devQKVProjArray),
                       static_cast<DT *>(m->keyCache),
                       static_cast<DT *>(m->valueCache),
                       m->token_infos,
                       m->request_infos,
                       m->beam_token_infos,
                       m->beam_request_infos,
                       m->causalMask,
                       m->qProjSize,
                       m->kProjSize,
                       m->vProjSize,
                       num_tokens,
                       BatchConfig::max_sequence_length() +
                           BatchConfig::max_spec_tree_token_num(),
                       /*root*/ curr_depth == 0,
                       m->hidden_size);
  }
}

template <typename DT>
__global__ void spec_fill_entries_above_diagonal(DT *matrix,
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
void compute_attention_kernel(SpecIncMultiHeadSelfAttentionMeta const *m,
                              BeamSearchBatchConfig const *bc,
                              int shard_id,
                              DT *output_ptr,
                              hipStream_t stream) {
  checkCUDA(hipblasSetStream(m->handle.blas, stream));
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));
  hipblasDatatype_t hipblas_data_type = ff_to_cuda_datatype(m->output_type[0]);
  miopenDataType_t miopen_data_type = ff_to_cudnn_datatype(m->output_type[0]);
  assert(data_type_size(m->output_type[0]) == sizeof(DT));
  hipblasDatatype_t compute_type = hipblas_data_type;

  int num_tokens = bc->num_active_tokens();
  int tokens_previous_requests = 0;
  int tokens_prev_requests_squares = 0;
  int q_block_size = m->qProjSize;

  int kt_block_size = m->kProjSize;
  int kt_req_block_size = kt_block_size * m->num_q_heads *
                          (BatchConfig::max_sequence_length() +
                           BatchConfig::max_spec_tree_token_num());
  int vt_block_size = m->vProjSize;
  int vt_req_block_size = vt_block_size * m->num_q_heads *
                          (BatchConfig::max_sequence_length() +
                           BatchConfig::max_spec_tree_token_num());
  assert(m->qProjSize == m->kProjSize);

  for (int i = 0; i < bc->max_requests_per_batch(); i++) {
    if (bc->request_completed[i] || (!bc->requestsInfo[i].prompt_phase) ||
        (bc->requestsInfo[i].num_tokens_in_batch == 0)) {
      continue;
    } else if (tokens_previous_requests < bc->num_generation_tokens) {
      tokens_previous_requests += bc->requestsInfo[i].num_tokens_in_batch;
      continue;
    }

    // all requests in prompt phase should only have one sub requests;
    assert(bc->sub_requests[i] == 1);
    // int num_new_tokens = bc->num_processing_tokens[i];
    // int total_tokens = bc->token_last_available_idx[i] + 1;

    int num_new_tokens = bc->requestsInfo[i].num_tokens_in_batch;
    int total_tokens = bc->requestsInfo[i].first_token_depth_in_request +
                       bc->requestsInfo[i].num_tokens_in_batch;

    if (num_new_tokens <= 0) {
      continue;
    }

    // Compute (QK^T/sqrt(d_k))
    int m_ = num_new_tokens;
    int n = total_tokens;
    int k = m->qProjSize;
    int lda = k * m->num_q_heads * QKV_WEIGHT_NUM, ldb = k * m->num_q_heads,
        ldc = m_;
    int strideA = q_block_size;
    int strideB = kt_block_size;
    int strideC = num_new_tokens * total_tokens;

    // a flag of using this scaling alpha
    DT alpha = 1.0f, beta = 0.0f;
    if (*m->qk_prod_scaling) {
      alpha = static_cast<DT>(1.0f / sqrt(m->kProjSize));
    }
    // To get A, skip over Q entries from previous requests (same head)
    DT const *A = static_cast<DT *>(m->devQKVProjArray) +
                  bc->requestsInfo[i].first_token_offset_in_batch *
                      m->qProjSize * m->num_q_heads * QKV_WEIGHT_NUM;
    DT const *B = static_cast<DT *>(m->keyCache) + i * kt_req_block_size;
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
      size_t parallelism = m->num_q_heads * total_tokens * num_new_tokens;
      hipLaunchKernelGGL(HIP_KERNEL_NAME(apply_position_bias_qkprd<DT>),
                         GET_BLOCKS(parallelism),
                         min((size_t)CUDA_NUM_THREADS, parallelism),
                         0,
                         stream,
                         C,
                         num_new_tokens,
                         total_tokens,
                         m->num_q_heads,
                         m->global_num_q_heads,
                         shard_id);
    }

    // Fill all elements above diagonal in qk prods with -inf to force
    // causal attention.
    assert(num_new_tokens <= total_tokens);
    if (num_new_tokens > 1) {
      size_t parallelism = m->num_q_heads * num_new_tokens * total_tokens;
      hipLaunchKernelGGL(HIP_KERNEL_NAME(spec_fill_entries_above_diagonal<DT>),
                         GET_BLOCKS(parallelism),
                         min((size_t)CUDA_NUM_THREADS, parallelism),
                         0,
                         stream,
                         C,
                         num_new_tokens,
                         total_tokens,
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
    int c_param = total_tokens;
    int h_param = 1;
    int w_param = num_new_tokens;
    checkCUDNN(miopenSet4dTensorDescriptor(
        m->qk_tensor, miopen_data_type, n_param, c_param, h_param, w_param));
    float softmax_alpha = 1.0f, softmax_beta = 0.0f;
    DT *C_softmax = static_cast<DT *>(m->qk_prods_softmax) +
                    m->num_q_heads * tokens_prev_requests_squares;
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
    m_ = m->vProjSize;
    n = num_new_tokens;
    k = total_tokens;
    lda = m_ * m->num_q_heads, ldb = n, ldc = m_ * m->num_q_heads;
    strideA = vt_block_size;
    strideB = num_new_tokens * total_tokens;
    strideC = m->vProjSize;
    // To get A, skip over V^T entries from previous requests (all heads +
    // padding)
    A = static_cast<DT *>(m->valueCache) + i * vt_req_block_size;
    // To get B, skip over softmax(QK^T/sqrt(d_k)) entries from previous
    // requests (all heads)
    B = C_softmax;
    // To get C, skip over softmax(QK^T/sqrt(d_k))V products from previous
    // requests

    int token_offset = bc->requestsInfo[i].first_token_offset_in_batch;

    C = static_cast<DT *>(m->attn_heads) +
        (token_offset)*m->num_q_heads * m->vProjSize;
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

    tokens_previous_requests += num_new_tokens;
    tokens_prev_requests_squares += num_new_tokens * total_tokens;
  }

  if (tokens_previous_requests != (num_tokens - bc->num_generation_tokens)) {
    bc->print();
    printf("tokens_previous_requests: %i\n", tokens_previous_requests);
    printf("num_tokens: %i\n", num_tokens);
    printf("bc->num_generation_tokens: %i\n", bc->num_generation_tokens);
  }
  assert(tokens_previous_requests == (num_tokens - bc->num_generation_tokens));
}

template <typename DT>
void inference_kernel(SpecIncMultiHeadSelfAttentionMeta const *m,
                      BeamSearchBatchConfig const *bc,
                      int shard_id,
                      DT const *qkv_ptr,
                      DT const *weight_ptr,
                      DT *output_ptr,
                      hipStream_t stream) {

  // phase 0: copy calculated qkv into devQKVProjArray
  // [qProjSize, num_heads, 3, num_new_tokens]
  size_t qkv_proj_size =
      m->qProjSize * m->num_q_heads * QKV_WEIGHT_NUM * bc->num_active_tokens();

  hipMemcpyAsync(m->devQKVProjArray,
                 qkv_ptr,
                 qkv_proj_size *
                     sizeof(DT), // is this right, do we need layers etc here
                 hipMemcpyDeviceToDevice,
                 stream);
  // phase 1: Implement kernel to compute KQV for input tokens
  // TODO WARNING: this is commented out only because we are fixing the inc_attn
  // first
  compute_qkv_kernel(
      m, bc, shard_id, static_cast<DT *>(m->devQKVProjArray), stream);
  // phase 2: Update key/val cache
  update_kv_cache_kernel<DT>(m, bc, stream);
  if (bc->num_generation_tokens > 0) {
    compute_attention_kernel<DT>(
        m, bc, static_cast<DT *>(m->attn_heads), stream);
  }
  // phase 3: Compute attention score
  // 3 kernels for pahse 3: matmul1 - softmax - matmal2
  if (bc->num_tokens > bc->num_generation_tokens) {
    compute_attention_kernel(m, bc, shard_id, output_ptr, stream);
  }

  int num_tokens = bc->num_active_tokens();

  hipMemcpyAsync(output_ptr,
                 m->attn_heads,
                 m->oProjSize * num_tokens * sizeof(DT),
                 hipMemcpyDeviceToDevice,
                 stream);
}

} // namespace SpecIncMultiHeadSelfAttention
} // namespace Kernels

/*static*/
void SpecIncMultiHeadSelfAttention::inference_kernel_wrapper(
    SpecIncMultiHeadSelfAttentionMeta const *m,
    BeamSearchBatchConfig const *bc,
    int shard_id,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &output) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
  }

  assert(input.data_type == output.data_type);

  if (input.data_type == DT_HALF) {
    Kernels::SpecIncMultiHeadSelfAttention::inference_kernel(
        m, bc, shard_id, input.get_half_ptr(), output.get_half_ptr(), stream);
  } else if (input.data_type == DT_FLOAT) {
    Kernels::SpecIncMultiHeadSelfAttention::inference_kernel(
        m, bc, shard_id, input.get_float_ptr(), output.get_float_ptr(), stream);
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
    printf("SpecIncMultiHeadSelfAttention forward time = %.2fms\n", elapsed);
  }
}

SpecIncMultiHeadSelfAttentionMeta::SpecIncMultiHeadSelfAttentionMeta(
    FFHandler handler,
    SpecIncMultiHeadSelfAttention const *attn,
    MemoryAllocator &gpu_mem_allocator,
    int num_samples,
    int _num_q_heads,
    int _num_kv_heads)
    : IncMultiHeadSelfAttentionMeta(handler,
                                    BEAM_SEARCH_MODE,
                                    attn,
                                    attn->qSize,
                                    attn->kSize,
                                    attn->vSize,
                                    attn->qProjSize,
                                    attn->kProjSize,
                                    attn->vProjSize,
                                    attn->oProjSize,
                                    attn->rotary_embedding_meta,
                                    attn->scaling_query,
                                    attn->qk_prod_scaling,
                                    attn->position_bias,
                                    attn->scaling_factor,
                                    gpu_mem_allocator,
                                    num_samples,
                                    attn->num_q_heads,
                                    attn->num_kv_heads,
                                    _num_q_heads,
                                    _num_kv_heads,
                                    DT_NONE,
                                    false) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDNN(miopenSetStream(handler.dnn, stream));

  // allocate memory for the seqArray and reserve space
  {
    beam_token_infos =
        static_cast<BeamSearchBatchConfig::BeamSearchPerTokenInfo *>(
            handler.batch_config_metadata->beamTokenInfo);
    beam_request_infos =
        static_cast<BeamSearchBatchConfig::BeamSearchPerRequestInfo *>(
            handler.batch_config_metadata->beamRequestsInfo);
    causalMask = static_cast<BatchConfig::BitMask *>(
        handler.batch_config_metadata->causalMask);
    request_completed =
        static_cast<bool *>(handler.batch_config_metadata->request_completed);
  }

  checkCUDA(hipStreamSynchronize(stream));
}

SpecIncMultiHeadSelfAttentionMeta::~SpecIncMultiHeadSelfAttentionMeta(void) {
  if (beam_search_reserve_inst != Realm::RegionInstance::NO_INST) {
    beam_search_reserve_inst.destroy();
  }
}

}; // namespace FlexFlow

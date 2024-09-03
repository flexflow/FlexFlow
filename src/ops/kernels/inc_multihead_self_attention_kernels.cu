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
#include "flexflow/batch_config.h"
#include <cassert>
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include "cuComplex.h"
#endif
#include "flashinfer/pos_enc.cuh"
#include "flexflow/attention_config.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/inc_multihead_self_attention.h"
#include "flexflow/ops/kernels/decompress_kernels.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_kernels.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_utils.cuh"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;

using flashinfer::BatchQKApplyLlama31Rotary;
using flashinfer::BatchQKApplyRotary;

#define WARP_SIZE 32

namespace Kernels {
namespace IncMultiHeadAttention {

// only used by MPT model. https://arxiv.org/abs/2108.12409
template <typename DT>
__global__ void apply_position_bias_qkprd(DT *input_ptr,
                                          int num_tokens,
                                          int num_total_tokens,
                                          int num_heads,
                                          int global_num_q_heads,
                                          int shard_id) {
  CUDA_KERNEL_LOOP(i, num_tokens * num_total_tokens * num_heads) {
    // get head_idx,
    int head_idx = i / (num_tokens * num_total_tokens) + (num_heads * shard_id);
    int position_idx = (i / num_tokens) % num_total_tokens;
    position_idx = position_idx + 1 - num_total_tokens;
    // 8 is alibi_bias_max in
    // https://huggingface.co/mosaicml/mpt-30b/blob/main/config.json
    float base = (float)(head_idx + 1) * 8 / global_num_q_heads;
    float slopes = 1.0 / pow(2, base);
    // if(i == 0){
    //   printf("see position: %d, %f, %f, %f\n", position_idx, base, slopes,
    //   position_idx * slopes);
    // }
    input_ptr[i] += static_cast<DT>(position_idx * slopes);
  }
}

template <typename DT>
__global__ void apply_proj_bias_w(DT *input_ptr,
                                  DT const *bias_ptr,
                                  int num_tokens,
                                  int qkv_weight_size,
                                  int o_dim) {
  CUDA_KERNEL_LOOP(i, num_tokens * o_dim) {
    int bias_idx = qkv_weight_size + i % o_dim;
    input_ptr[i] += bias_ptr[bias_idx];
  }
}

template <typename DT>
__global__ void apply_proj_bias_qkv(DT *input_ptr,
                                    DT const *bias_ptr,
                                    int shard_id,
                                    int num_tokens,
                                    int qk_dim,
                                    int v_dim,
                                    int global_num_q_heads,
                                    int num_q_heads,
                                    bool scaling_query,
                                    float scaling_factor,
                                    int hidden_size) {
  CUDA_KERNEL_LOOP(i, num_tokens * hidden_size * QKV_WEIGHT_NUM) {
    // for simplicity, assume q, k, v is in same shape
    // 0->q, 1->k, 2->v
    // int qkv_index = i / (num_tokens * qk_dim) % 3;

    int token_idx = i / (hidden_size * QKV_WEIGHT_NUM);
    size_t in_token_idx = i - token_idx * hidden_size * QKV_WEIGHT_NUM;

    int qkv_index = in_token_idx / hidden_size;

    int proj_size = qkv_index == 0 ? qk_dim : qk_dim;

    int head_idx =
        (in_token_idx - qkv_index * num_q_heads * proj_size) / proj_size;
    int global_head_idx = head_idx + shard_id * num_q_heads;

    size_t pre_length =
        qkv_index == 0
            ? 0
            : (qkv_index == 1 ? qk_dim * global_num_q_heads
                              : qk_dim * global_num_q_heads * KV_WEIGHT_NUM);

    size_t bias_idx = pre_length + global_head_idx * proj_size + i % proj_size;

    input_ptr[i] += bias_ptr[bias_idx];

    if (scaling_query && qkv_index == 0) {
      input_ptr[i] *= scaling_factor;
    }
  }
}

template <typename DT>
__global__ void scaling_query_kernel(DT *input_ptr,
                                     int qk_dim,
                                     int num_tokens,
                                     int num_q_heads,
                                     float scaling_factor,
                                     int hidden_size) {
  CUDA_KERNEL_LOOP(i, num_tokens * hidden_size) {
    int token_idx = i / hidden_size;
    input_ptr[i % hidden_size + token_idx * hidden_size * QKV_WEIGHT_NUM] *=
        scaling_factor;
  }
}

template <typename DT>
__global__ void
    apply_rotary_embedding_native(DT *input_ptr,
                                  cuFloatComplex *complex_input,
                                  BatchConfig::PerTokenInfo const *tokenInfos,
                                  int qk_dim,
                                  int num_q_heads,
                                  int num_tokens,
                                  int num_kv_heads,
                                  int q_block_size,
                                  int k_block_size,
                                  int q_array_size) {
  CUDA_KERNEL_LOOP(
      i, num_tokens * (qk_dim * num_q_heads + qk_dim * num_kv_heads) / 2) {
    // create complex number
    bool q_tensor = i < (q_array_size / 2);
    int proj_size = q_tensor ? qk_dim : qk_dim;
    int real_i = q_tensor ? i : i - q_array_size / 2;

    int head_idx = real_i / (num_tokens * proj_size / 2);
    int idx = real_i % (num_tokens * proj_size / 2);
    int real_part_index = idx * 2 +
                          head_idx * (q_tensor ? q_block_size : k_block_size) +
                          (q_tensor ? 0 : q_array_size);

    int complex_part_index = real_part_index + 1;

    complex_input[i] = {input_ptr[real_part_index],
                        input_ptr[complex_part_index]};

    int token_idx =
        (real_i - head_idx * (num_tokens * proj_size / 2)) / (proj_size / 2);
    size_t pos = tokenInfos[token_idx].abs_depth_in_request;

    // float before_real = complex_input[i].x, before_complex =
    // complex_input[i].y;

    int pos_i = real_i % (proj_size / 2);
    float freq = pos * (1.0 / pow(10000.0, (float)2 * pos_i / proj_size));
    cuFloatComplex complex_pos = {cos(freq), sin(freq)};

    complex_input[i] = cuCmulf(complex_input[i], complex_pos);
    input_ptr[real_part_index] = complex_input[i].x;
    input_ptr[complex_part_index] = complex_input[i].y;
  }
}

template <typename DT>
void compute_qkv(IncMultiHeadSelfAttentionMeta const *m,
                 BatchConfig const *bc,
                 int shard_id,
                 DT const *input_ptr,
                 DT const *weight_ptr,
                 DT *output_ptr,
                 DT const *bias_ptr,
                 cudaStream_t stream) {

  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  cudaDataType_t cublas_data_type = ff_to_cuda_datatype(m->output_type[0]);
#if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
  cudaDataType_t compute_type = cublas_data_type;
#else
  // For best performance, set the default cublas compute type to
  // CUBLAS_COMPUTE_16F for half precision and to
  // CUBLAS_COMPUTE_32F_FAST_16F for full precision
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
  if (m->output_type[0] == DT_FLOAT) {
    compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
  }
#endif

  //   int device;
  //   checkCUDA(cudaGetDevice(&device));
  //   cudaEvent_t t_start, t_end;
  //   checkCUDA(cudaEventCreate(&t_start));
  //   checkCUDA(cudaEventCreate(&t_end));
  //   checkCUDA(cudaEventRecord(t_start, stream));

  // Step 1: Compute QKV projections
  {
    DT alpha = 1.0f, beta = 0.0f;
    // after transpositions
    int m_q = m->qk_dim * m->num_q_heads;
    int m_k = m->qk_dim * m->num_q_heads;
    int m_v = m->v_dim * m->num_q_heads;
    assert(m_q == m_k && m_k == m_v); // keep things simple for now
    int n = bc->num_active_tokens();
    int k = m->hidden_size;
    int m_ = m_q * QKV_WEIGHT_NUM;
    // before transpositions
    int lda = k, ldb = k, ldc = m_;
    // matrix A: QKV weights
    // matrix A's layout: [hidden_size (hidden_dim), qk_dim, num_heads, 3]
    // matrix B: input
    // matrix B's layout: [hidden_size (hidden_dim), num_new_tokens]
    // matrix C: devQKVProjArray
    // matrix B's layout: [qk_dim, num_heads, 3, num_new_tokens]
    checkCUDA(cublasGemmEx(m->handle.blas,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           m_,
                           n,
                           k,
                           &alpha,
                           weight_ptr,
                           cublas_data_type,
                           lda,
                           input_ptr,
                           cublas_data_type,
                           ldb,
                           &beta,
                           output_ptr,
                           cublas_data_type,
                           ldc,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

  //   checkCUDA(cudaEventRecord(t_end, stream));
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   float elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
  //   if (bc->inference_mode == TREE_VERIFY_MODE and device == 0) {
  //     std::cout << "GEMM time: " << elapsed << " ms\n";
  //   }

  int num_tokens = bc->num_active_tokens();
  int parallelism = m->qk_dim * num_tokens * m->num_q_heads;

  // Step 2: apply bias for QKV, or scale the query
  if (*m->qkv_bias) {
    apply_proj_bias_qkv<<<GET_BLOCKS(parallelism),
                          min(CUDA_NUM_THREADS, parallelism),
                          0,
                          stream>>>(output_ptr,
                                    bias_ptr,
                                    shard_id,
                                    num_tokens,
                                    m->qk_dim,
                                    m->v_dim,
                                    m->global_num_q_heads,
                                    m->num_q_heads,
                                    *m->scaling_query,
                                    m->scaling_factor,
                                    m->local_hidden_size);
  } else if (m->scaling_query) {
    scaling_query_kernel<<<GET_BLOCKS(parallelism),
                           min(CUDA_NUM_THREADS, parallelism),
                           0,
                           stream>>>(output_ptr,
                                     num_tokens,
                                     m->num_q_heads,
                                     m->qk_dim,
                                     m->scaling_factor,
                                     m->local_hidden_size);
  }
}

template <typename DT>
__global__ void apply_pos_encoding_to_tokens_in_batch_kernel(
    DT *input_ptr,
    BatchConfig::PerTokenInfo const *tokenInfos,
    int qk_dim,
    int num_tokens,
    size_t q_array_size,
    int hidden_size) {
  CUDA_KERNEL_LOOP(i, num_tokens * hidden_size) {
    // create complex number
    bool q_tensor = i < (q_array_size / 2);
    int proj_size = q_tensor ? qk_dim : qk_dim;
    int real_i = q_tensor ? i : i - q_array_size / 2;

    int token_idx = real_i / (hidden_size / 2);
    int idx = real_i % (proj_size / 2);
    int head_idx = (real_i - (token_idx * (hidden_size / 2))) / (proj_size / 2);

    int real_part_index = idx + head_idx * proj_size +
                          token_idx * hidden_size * QKV_WEIGHT_NUM +
                          hidden_size * (q_tensor ? 0 : 1);
    int complex_part_index = real_part_index + (proj_size / 2);

    cuFloatComplex cii = {input_ptr[real_part_index],
                          input_ptr[complex_part_index]};

    // get the freq_cis: shape 1 * (qk_dim/2) = 1 * 64
    // apply a Cartesian coordinate transformation
    // multiple with input & /copy back to q/k

    // get position of token

    size_t pos = tokenInfos[token_idx].abs_depth_in_request;

    float freq = pos * (1.0 / pow(10000.0, (float)2 * idx / proj_size));
    cuFloatComplex complex_pos = {cos(freq), sin(freq)};

    cii = cuCmulf(cii, complex_pos);
    input_ptr[real_part_index] = cii.x;
    input_ptr[complex_part_index] = cii.y;
  }
}

template <typename DT>
void apply_pos_encoding_to_tokens_in_batch(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    DT *output_ptr,
    cudaStream_t stream) {
  // apply rotary embedding if needed
  if (!*m->apply_rotary_embedding) {
    return;
  }
  int num_tokens = bc->num_active_tokens();
  int parallelism = num_tokens * m->local_hidden_size;
  size_t q_array_size = m->qk_dim * num_tokens * m->num_q_heads;
  apply_pos_encoding_to_tokens_in_batch_kernel<<<GET_BLOCKS(parallelism),
                                                 min(CUDA_NUM_THREADS,
                                                     parallelism),
                                                 0,
                                                 stream>>>(
      output_ptr,
      m->token_infos,
      m->qk_dim,
      num_tokens,
      q_array_size,
      m->local_hidden_size);
}

__global__ void apply_pos_encoding_to_streaming_proj_kernel(
    half *kv_cache,
    BatchConfig::PerRequestInfo const *requestInfos,
    bool const *request_available,
    int const max_num_pages,
    int num_kv_heads,
    int head_dim,
    StreamingCacheInfo const *streaming_cache_infos,
    uint32_t const max_num_requests) {
  int const kv_hidden_size = num_kv_heads * head_dim;
  int const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int token_idx = thread_idx / (kv_hidden_size / 2);
  // Each complex is consist of (i, i + head_dim / 2) wuthin the same head.
  int const head_idx = (thread_idx % (kv_hidden_size / 2)) / (head_dim / 2);
  int const offset_in_head = thread_idx % (head_dim / 2);
  // Get the corresponding request index and token index in the request.
  int request_idx = 0;
  while (token_idx >= 0 && request_idx < max_num_requests) {
    if (request_available[request_idx]) {
      token_idx -= streaming_cache_infos[request_idx].commit_len;
    }
    request_idx++;
  }
  if (token_idx >= 0) {
    return;
  }
  request_idx--;
  token_idx += streaming_cache_infos[request_idx].commit_len;

  // Get the real and complex part index for the current complex.
  int const real_part_idx =
      get_k_entry_offset(
          request_idx, token_idx, max_num_pages, num_kv_heads, head_dim) +
      head_idx * head_dim + offset_in_head;
  int const complex_part_idx = real_part_idx + head_dim / 2;

  // Apply the rotary position encoding.
  cuFloatComplex cii = {kv_cache[real_part_idx], kv_cache[complex_part_idx]};
  size_t pos = token_idx;
  float freq = pos * (1.0 / pow(10000.0, (float)2 * offset_in_head / head_dim));
  cuFloatComplex complex_pos = {cos(freq), sin(freq)};
  cii = cuCmulf(cii, complex_pos);
  kv_cache[real_part_idx] = cii.x;
  kv_cache[complex_part_idx] = cii.y;
}

template <typename DT>
void apply_pos_encoding_to_streaming_proj(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    cudaStream_t stream) {
  assert(m->streaming_cache);
  int const kv_hidden_size = m->num_kv_heads * m->qk_dim;
  int num_tokens = 0;
  for (int req_idx = 0; req_idx < BatchConfig::max_requests_per_batch();
       req_idx++) {
    if (!bc->request_available[req_idx]) {
      continue;
    }
    num_tokens += bc->streamingCacheInfo[req_idx].commit_len;
  }
  int parallelism = num_tokens * kv_hidden_size / 2;
  int const max_num_pages = round_up_pages(
      BatchConfig::MAX_STREAMING_POS - BatchConfig::get_max_tree_depth() +
      BatchConfig::max_spec_tree_token_num());
  apply_pos_encoding_to_streaming_proj_kernel<<<GET_BLOCKS(parallelism),
                                                min(CUDA_NUM_THREADS,
                                                    parallelism),
                                                0,
                                                stream>>>(
      static_cast<half *>(m->kvCache),
      m->request_infos,
      m->request_available,
      max_num_pages,
      m->num_kv_heads,
      m->qk_dim,
      m->streaming_cache_infos,
      bc->max_requests_per_batch());
}

template <typename DT>
__global__ void
    update_qkv_in_batch_kernel(DT *qkv_proj_array,
                               half *qTmp_ptr,
                               half *kvCache_ptr,
                               BatchConfig::PerTokenInfo const *tokenInfos,
                               int const max_num_pages,
                               int num_q_heads,
                               int num_kv_heads,
                               int head_dim,
                               int num_new_tokens) {
  int const q_hidden_size = num_q_heads * head_dim;
  int const temp_kv_hidden_size = num_q_heads * head_dim; // temporary hard code
  int const kv_hidden_size = num_kv_heads * head_dim;
  int const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int const token_idx = thread_idx / q_hidden_size;
  int const offset = thread_idx % q_hidden_size;
  if (token_idx >= num_new_tokens) {
    return;
  }

  int const req_idx = tokenInfos[token_idx].request_index;
  int token_abs_idx = tokenInfos[token_idx].abs_index_in_request;

  size_t from_idx = token_idx * (q_hidden_size + temp_kv_hidden_size * 2);
  qTmp_ptr[token_idx * q_hidden_size + offset] =
      static_cast<half>(qkv_proj_array[from_idx + offset]);

  if (offset < kv_hidden_size) {
    size_t to_k_idx = get_k_entry_offset(
               req_idx, token_abs_idx, max_num_pages, num_kv_heads, head_dim),
           to_v_idx = get_v_entry_offset(
               req_idx, token_abs_idx, max_num_pages, num_kv_heads, head_dim);
    // key and value cache should be stored interleaved
    int const stride = num_q_heads / num_kv_heads;
    int const kv_offset =
        offset / head_dim * stride * head_dim + offset % head_dim;
    kvCache_ptr[to_k_idx + offset] =
        static_cast<half>(qkv_proj_array[from_idx + q_hidden_size + kv_offset]);
    kvCache_ptr[to_v_idx + offset] =
        static_cast<half>(qkv_proj_array[from_idx + q_hidden_size +
                                         temp_kv_hidden_size + kv_offset]);
  }
}

template <typename DT>
void update_qkv_in_batch(IncMultiHeadSelfAttentionMeta const *m,
                         BatchConfig const *bc,
                         cudaStream_t stream) {
  int num_new_tokens = bc->num_active_tokens();
  int parallelism = m->local_hidden_size * num_new_tokens;
  int const max_num_pages =
      round_up_pages(BatchConfig::max_sequence_length() +
                     BatchConfig::max_spec_tree_token_num());
  update_qkv_in_batch_kernel<<<GET_BLOCKS(parallelism),
                               min(CUDA_NUM_THREADS, parallelism),
                               0,
                               stream>>>(static_cast<DT *>(m->devQKVProjArray),
                                         static_cast<half *>(m->queryTmp),
                                         static_cast<half *>(m->kvCache),
                                         m->token_infos,
                                         max_num_pages,
                                         m->num_q_heads,
                                         m->num_kv_heads,
                                         m->qk_dim,
                                         num_new_tokens);
}

__global__ void update_kv_in_streaming_cache_kernel(
    half *pre_pos_enc_buf,
    half *kv_cache,
    BatchConfig::PerRequestInfo const *requestInfos,
    bool const *request_available,
    int const max_num_pages_pre_pos_enc_buf,
    int const max_num_pages_kv_cache,
    int num_kv_heads,
    int head_dim,
    StreamingCacheInfo const *streaming_cache_infos,
    uint32_t const max_num_requests) {
  int const kv_hidden_size = num_kv_heads * head_dim;
  int const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int token_idx = thread_idx / kv_hidden_size;
  int const offset = thread_idx % kv_hidden_size;
  int request_idx = 0;
  while (token_idx >= 0 && request_idx < max_num_requests) {
    if (request_available[request_idx]) {
      token_idx -= streaming_cache_infos[request_idx].commit_len;
    }
    request_idx++;
  }
  if (token_idx >= 0) {
    return;
  }
  request_idx--;
  token_idx += streaming_cache_infos[request_idx].commit_len;

  size_t from_k_idx = get_k_entry_offset(request_idx,
                                         token_idx,
                                         max_num_pages_pre_pos_enc_buf,
                                         num_kv_heads,
                                         head_dim),
         from_v_idx = get_v_entry_offset(request_idx,
                                         token_idx,
                                         max_num_pages_pre_pos_enc_buf,
                                         num_kv_heads,
                                         head_dim);

  // to_idx should consider the rolling property of the window cache
  int to_idx = token_idx;
  StreamingCacheInfo const &info = streaming_cache_infos[request_idx];
  if (info.commit_len >= info.sink_cache_size + info.window_cache_size &&
      to_idx >= info.sink_cache_size) {
    to_idx -= info.sink_cache_size;
    to_idx = (to_idx + info.window_cache_size - info.window_back) %
             info.window_cache_size;
    to_idx += info.sink_cache_size;
  }

  size_t to_k_idx = get_k_entry_offset(request_idx,
                                       to_idx,
                                       max_num_pages_kv_cache,
                                       num_kv_heads,
                                       head_dim),
         to_v_idx = get_v_entry_offset(request_idx,
                                       to_idx,
                                       max_num_pages_kv_cache,
                                       num_kv_heads,
                                       head_dim);

  kv_cache[to_k_idx + offset] = pre_pos_enc_buf[from_k_idx + offset];
  kv_cache[to_v_idx + offset] = pre_pos_enc_buf[from_v_idx + offset];
}

template <typename DT>
void update_kv_in_streaming_cache(IncMultiHeadSelfAttentionMeta const *m,
                                  BatchConfig const *bc,
                                  cudaStream_t stream) {
  assert(m->streaming_cache);
  int const kv_hidden_size = m->num_kv_heads * m->qk_dim;
  int num_tokens = 0;
  for (int req_idx = 0; req_idx < BatchConfig::max_requests_per_batch();
       req_idx++) {
    if (!bc->request_available[req_idx]) {
      continue;
    }
    num_tokens += bc->streamingCacheInfo[req_idx].commit_len;
  }
  int parallelism = kv_hidden_size * num_tokens;
  int const max_num_pages_pre_pos_enc_buf = round_up_pages(
      BatchConfig::MAX_STREAMING_POS - BatchConfig::get_max_tree_depth());
  int const max_num_pages_kv_cache = round_up_pages(
      BatchConfig::MAX_STREAMING_POS - BatchConfig::get_max_tree_depth() +
      BatchConfig::max_spec_tree_token_num());

  update_kv_in_streaming_cache_kernel<<<GET_BLOCKS(parallelism),
                                        min(CUDA_NUM_THREADS, parallelism),
                                        0,
                                        stream>>>(
      static_cast<half *>(m->streamingPrePosEncBuf),
      static_cast<half *>(m->kvCache),
      m->request_infos,
      m->request_available,
      max_num_pages_pre_pos_enc_buf,
      max_num_pages_kv_cache,
      m->num_kv_heads,
      m->qk_dim,
      m->streaming_cache_infos,
      bc->max_requests_per_batch());
}

template <typename DT>
__global__ void
    commit_kv_kernel(DT const *qkv_proj_array,
                     half *pre_pos_enc_buf,
                     BatchConfig::PerTokenInfo const *tokenInfos,
                     BatchConfig::PerRequestInfo const *requestInfos,
                     int const max_num_pages,
                     int num_q_heads,
                     int num_kv_heads,
                     int head_dim,
                     StreamingCacheInfo const *streaming_cache_infos,
                     int num_new_tokens) {
  int const q_hidden_size = num_q_heads * head_dim;
  int const temp_kv_hidden_size = num_q_heads * head_dim; // temporary hard code
  int const kv_hidden_size = num_kv_heads * head_dim;
  int const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int const token_idx = thread_idx / kv_hidden_size;
  int const offset = thread_idx % kv_hidden_size;
  if (token_idx >= num_new_tokens) {
    return;
  }
  int const request_idx = tokenInfos[token_idx].request_index;

  StreamingCacheInfo const &info = streaming_cache_infos[request_idx];
  int to_idx = tokenInfos[token_idx].abs_index_in_request;
  // cases that get over the boundary:
  // 1. commit_len < sink_cache_size: commit to sink, window, window_back is
  // after commit_len.
  // 2. sink_cache_size <= commit_len < sink_cache_size + window_cache_size:
  // commit to window, window_back + sink_cache_size = commit_len, pointing to
  // the same position.
  // 3. commit_len >= sink_cache_size + window_cache_size: commit to window,
  // window is full before this commit, window_back is pointing to the real
  // position.
  if (to_idx >= info.sink_cache_size + info.window_cache_size) {
    to_idx = to_idx - info.commit_len + info.window_back;
    if (info.commit_len < info.sink_cache_size) {
      // For case 1, compensating for sink offset, because window_back is
      // someway back from commit_len.
      to_idx -= info.sink_cache_size - info.commit_len;
    }
    to_idx = info.sink_cache_size + to_idx % info.window_cache_size;
  }
  // TODO: For now don't consider the case that the commit tokens roll over the
  // for more than once. In this case, we should only count the last tokens in
  // the same window position.

  size_t from_idx = token_idx * (q_hidden_size + temp_kv_hidden_size * 2);
  size_t to_k_idx = get_k_entry_offset(
             request_idx, to_idx, max_num_pages, num_kv_heads, head_dim),
         to_v_idx = get_v_entry_offset(
             request_idx, to_idx, max_num_pages, num_kv_heads, head_dim);

  pre_pos_enc_buf[to_k_idx + offset] =
      static_cast<half>(qkv_proj_array[from_idx + q_hidden_size + offset]);
  pre_pos_enc_buf[to_v_idx + offset] = static_cast<half>(
      qkv_proj_array[from_idx + q_hidden_size + temp_kv_hidden_size + offset]);
}

template <typename DT>
void commit_kv(IncMultiHeadSelfAttentionMeta const *m,
               BatchConfig const *bc,
               cudaStream_t stream) {
  assert(m->streaming_cache);
  int const kv_hidden_size = m->num_kv_heads * m->qk_dim;
  int const num_new_tokens = bc->num_active_tokens();
  int parallelism = kv_hidden_size * num_new_tokens;
  int const max_num_pages = round_up_pages(BatchConfig::MAX_STREAMING_POS -
                                           BatchConfig::get_max_tree_depth());

  commit_kv_kernel<<<GET_BLOCKS(parallelism),
                     min(CUDA_NUM_THREADS, parallelism),
                     0,
                     stream>>>(static_cast<DT *>(m->devQKVProjArray),
                               static_cast<half *>(m->streamingPrePosEncBuf),
                               m->token_infos,
                               m->request_infos,
                               max_num_pages,
                               m->num_q_heads,
                               m->num_kv_heads,
                               m->qk_dim,
                               m->streaming_cache_infos,
                               num_new_tokens);
}

template <typename DT>
__global__ void produce_output_kernel(half const *input_ptr,
                                      DT *output_ptr,
                                      int parallelism) {
  CUDA_KERNEL_LOOP(idx, parallelism) {
    output_ptr[idx] = static_cast<DT>(input_ptr[idx]);
  }
}

template <typename DT>
void produce_output(IncMultiHeadSelfAttentionMeta const *m,
                    BatchConfig const *bc,
                    DT *output_ptr,
                    cudaStream_t stream) {
  int parallelism = m->v_dim * m->num_q_heads * bc->num_active_tokens();
  produce_output_kernel<<<GET_BLOCKS(parallelism),
                          min(CUDA_NUM_THREADS, parallelism),
                          0,
                          stream>>>(m->outputTmp, output_ptr, parallelism);
}

template <typename DT>
void compute_o_prod_bias(IncMultiHeadSelfAttentionMeta const *m,
                         BatchConfig const *bc,
                         int shard_id,
                         DT *output_ptr,
                         DT const *weight_ptr,
                         DT const *bias_ptr,
                         int num_tokens,
                         cudaStream_t stream) {
  cudaDataType_t cublas_data_type = ff_to_cuda_datatype(m->output_type[0]);
  cudnnDataType_t cudnn_data_type = ff_to_cudnn_datatype(m->output_type[0]);
  assert(data_type_size(m->output_type[0]) == sizeof(DT));
#if CUDA_VERSION >= 11000
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best
  // performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
  cudaDataType_t compute_type = cublas_data_type;
#endif
  // Project to output, save result directly on output tensor
  {
    DT alpha = 1.0f, beta = 0.0f;
    // after transpositions
    int m_ = m->o_dim;
    int k = m->v_dim * m->num_q_heads;
    int n = num_tokens;
    // before transpositions
    int lda = k, ldb = k, ldc = m_;
    // matrix A: output projection weight
    // matrix A's layout: [v_dim * num_heads, o_dim]
    DT const *A = weight_ptr + m->hidden_size * (m->qk_dim * m->num_q_heads +
                                                 m->qk_dim * m->num_q_heads +
                                                 m->v_dim * m->num_q_heads);
    // matrix B: attn heads
    // matrix B's layout: [v_dim * num_heads, num_new_tokens]
    DT const *B = static_cast<DT *>(m->attn_heads);
    // matrix B: output
    // matrix B's layout: [o_dim, num_new_tokens]
    DT *C = static_cast<DT *>(output_ptr);

    checkCUDA(cublasGemmEx(m->handle.blas,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
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
  }
  // Add final output bias
  if (*m->final_bias && shard_id == 0) {
    int parallelism = m->o_dim * num_tokens;
    int qkv_weight_size = m->qk_dim * m->global_num_q_heads +
                          m->qk_dim * m->global_num_q_heads +
                          m->v_dim * m->global_num_q_heads;
    apply_proj_bias_w<<<GET_BLOCKS(parallelism),
                        min(CUDA_NUM_THREADS, parallelism),
                        0,
                        stream>>>(
        output_ptr, bias_ptr, num_tokens, qkv_weight_size, m->o_dim);
  }
}

template <typename DT>
void pre_build_weight(IncMultiHeadSelfAttentionMeta const *m,
                      GenericTensorAccessorR const weight,
                      DataType data_type,
                      cudaStream_t stream) {
  // additional processing for weight uploading
  // Note that we update weight_ptr and bias_ptr when uploading weight and
  // bias
  if (m->quantization_type != DT_NONE) {
    // copy weight_ptr to quantized_weight_ptr, do compression and store in
    // m->weight_ptr
    cudaMemcpyAsync(m->quantized_weight_ptr,
                    weight.get_byte_ptr(),
                    m->quantized_weightSize,
                    cudaMemcpyHostToDevice,
                    stream);

    if (m->quantization_type == DT_INT4) {
      int parallelism = m->qk_dim * m->hidden_size * m->num_q_heads / 2;
      decompress_int4_attention_weights<<<GET_BLOCKS(parallelism),
                                          min(CUDA_NUM_THREADS, parallelism),
                                          0,
                                          stream>>>(
          m->quantized_weight_ptr,
          static_cast<DT *>(m->weight_ptr),
          m->qk_dim,
          m->hidden_size,
          m->num_q_heads);
    } else {
      assert(m->quantization_type == DT_INT8);
      int parallelism = m->qk_dim * m->hidden_size * m->num_q_heads;
      decompress_int8_attention_weights<<<GET_BLOCKS(parallelism),
                                          min(CUDA_NUM_THREADS, parallelism),
                                          0,
                                          stream>>>(
          m->quantized_weight_ptr,
          static_cast<DT *>(m->weight_ptr),
          m->qk_dim,
          m->hidden_size,
          m->num_q_heads);
    }
  } else {
    if (data_type == DT_FLOAT) {
      cudaMemcpyAsync(m->weight_ptr,
                      weight.get_float_ptr(),
                      m->weightSize,
                      cudaMemcpyHostToDevice,
                      stream);
    } else if (data_type == DT_HALF) {
      cudaMemcpyAsync(m->weight_ptr,
                      weight.get_half_ptr(),
                      m->weightSize,
                      cudaMemcpyHostToDevice,
                      stream);
    } else {
      assert(false);
    }
  }
}

} // namespace IncMultiHeadAttention
} // namespace Kernels

using namespace Kernels::IncMultiHeadAttention;

template void Kernels::IncMultiHeadAttention::pre_build_weight<float>(
    IncMultiHeadSelfAttentionMeta const *m,
    GenericTensorAccessorR const weight,
    DataType data_type,
    cudaStream_t stream);

template void Kernels::IncMultiHeadAttention::pre_build_weight<half>(
    IncMultiHeadSelfAttentionMeta const *m,
    GenericTensorAccessorR const weight,
    DataType data_type,
    cudaStream_t stream);

template void Kernels::IncMultiHeadAttention::compute_qkv<float>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    int shard_id,
    float const *input_ptr,
    float const *weight_ptr,
    float *output_ptr,
    float const *bias_ptr,
    cudaStream_t stream);

template void Kernels::IncMultiHeadAttention::compute_qkv<half>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    int shard_id,
    half const *input_ptr,
    half const *weight_ptr,
    half *output_ptr,
    half const *bias_ptr,
    cudaStream_t stream);

template void
    Kernels::IncMultiHeadAttention::apply_pos_encoding_to_tokens_in_batch<
        float>(IncMultiHeadSelfAttentionMeta const *m,
               BatchConfig const *bc,
               float *output_ptr,
               cudaStream_t stream);

template void
    Kernels::IncMultiHeadAttention::apply_pos_encoding_to_tokens_in_batch<half>(
        IncMultiHeadSelfAttentionMeta const *m,
        BatchConfig const *bc,
        half *output_ptr,
        cudaStream_t stream);

template void
    Kernels::IncMultiHeadAttention::apply_pos_encoding_to_streaming_proj<float>(
        IncMultiHeadSelfAttentionMeta const *m,
        BatchConfig const *bc,
        cudaStream_t stream);

template void
    Kernels::IncMultiHeadAttention::apply_pos_encoding_to_streaming_proj<half>(
        IncMultiHeadSelfAttentionMeta const *m,
        BatchConfig const *bc,
        cudaStream_t stream);

template void Kernels::IncMultiHeadAttention::update_qkv_in_batch<float>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    cudaStream_t stream);

template void Kernels::IncMultiHeadAttention::update_qkv_in_batch<half>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    cudaStream_t stream);

template void
    Kernels::IncMultiHeadAttention::update_kv_in_streaming_cache<half>(
        IncMultiHeadSelfAttentionMeta const *m,
        BatchConfig const *bc,
        cudaStream_t stream);

template void
    Kernels::IncMultiHeadAttention::update_kv_in_streaming_cache<float>(
        IncMultiHeadSelfAttentionMeta const *m,
        BatchConfig const *bc,
        cudaStream_t stream);

template void Kernels::IncMultiHeadAttention::commit_kv<half>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    cudaStream_t stream);

template void Kernels::IncMultiHeadAttention::commit_kv<float>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    cudaStream_t stream);

template void Kernels::IncMultiHeadAttention::produce_output<float>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    float *output_ptr,
    cudaStream_t stream);

template void Kernels::IncMultiHeadAttention::produce_output<half>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    half *output_ptr,
    cudaStream_t stream);

template void Kernels::IncMultiHeadAttention::compute_o_prod_bias<float>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    int shard_id,
    float *output_ptr,
    float const *weight_ptr,
    float const *bias_ptr,
    int num_tokens,
    cudaStream_t stream);

template void Kernels::IncMultiHeadAttention::compute_o_prod_bias<half>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    int shard_id,
    half *output_ptr,
    half const *weight_ptr,
    half const *bias_ptr,
    int num_tokens,
    cudaStream_t stream);
}; // namespace FlexFlow

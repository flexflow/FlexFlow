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
#include "flashinfer/prefill_attention_decl.cuh"
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_kernels.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_utils.cuh"
#include "flexflow/ops/tree_inc_multihead_self_attention.h"
#include "flexflow/utils/cuda_helper.h"

#include <sstream>
#include <stdexcept>
#include <vector>

#define DISPATCH_GROUPSIZE(group_size, GROUP_SIZE, ...)                        \
  if (group_size == 1) {                                                       \
    constexpr size_t GROUP_SIZE = 1;                                           \
    __VA_ARGS__                                                                \
  } else if (group_size == 4) {                                                \
    constexpr size_t GROUP_SIZE = 4;                                           \
    __VA_ARGS__                                                                \
  } else if (group_size == 8) {                                                \
    constexpr size_t GROUP_SIZE = 8;                                           \
    __VA_ARGS__                                                                \
  } else {                                                                     \
    std::ostringstream err_msg;                                                \
    err_msg << "Unsupported group_size: " << group_size;                       \
    throw std::invalid_argument(err_msg.str());                                \
  }

#define DISPATCH_HEADDIM(head_dim, HEAD_DIM, ...)                              \
  switch (head_dim) {                                                          \
    case 64: {                                                                 \
      constexpr size_t HEAD_DIM = 64;                                          \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    case 128: {                                                                \
      constexpr size_t HEAD_DIM = 128;                                         \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    case 256: {                                                                \
      constexpr size_t HEAD_DIM = 256;                                         \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    default: {                                                                 \
      std::ostringstream err_msg;                                              \
      err_msg << "Unsupported head_dim: " << head_dim;                         \
      throw std::invalid_argument(err_msg.str());                              \
    }                                                                          \
  }

#define DISPATCH_PAGESIZE(page_size, PAGE_SIZE, ...)                           \
  if (page_size == kPagesize) {                                                \
    constexpr size_t PAGE_SIZE = kPagesize;                                    \
    __VA_ARGS__                                                                \
  } else {                                                                     \
    std::ostringstream err_msg;                                                \
    err_msg << "Unsupported page_size: " << page_size;                         \
    throw std::invalid_argument(err_msg.str());                                \
  }

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;

#define WARP_SIZE 32

using namespace Kernels::IncMultiHeadAttention;

namespace Kernels {
namespace TreeIncMultiHeadAttention {

using flashinfer::BatchPrefillHandler;
using flashinfer::BatchPrefillWithPagedKVCacheWrapperDispatched;
using flashinfer::MaskMode;
using flashinfer::paged_kv_t;
using flashinfer::PageStorage;
using flashinfer::PosEncodingMode;
using flashinfer::QKVLayout;

// page_idx: the assigned physical block index for this token
// token_idx: can be absolute index in the sequence but in there we just use it as an offset
// hidden_size: the size of the hidden dimension
__device__ __forceinline__ size_t get_k_entry_offset(int const token_idx,
                                                     int const page_idx,
                                                     int const hidden_size) {
  // page attention: changed
  size_t index = ((page_idx) * kPagesize * 2 + (token_idx % kPagesize)) * hidden_size;
  return index;
}

__device__ __forceinline__ size_t get_v_entry_offset(int const token_idx,
                                                     int const page_idx,
                                                     int const hidden_size) {
  size_t index = ((page_idx) * kPagesize * 2 + kPagesize + (token_idx % kPagesize)) * hidden_size;
  return index;
}

__global__ void commit_tokens_kernel(
    half *kCache_ptr,
    int32_t *kv_indptr,
    int32_t *kv_page_indices,
    BatchConfig::CommittedTokensInfo const *committedTokenInfos,
    bool const *request_available,
    int num_requests,
    int hidden_size,
    int num_committed_tokens,
    int const max_num_pages) {
  int const idx = blockIdx.x * blockDim.x + threadIdx.x;
  int const request_compact_idx = idx / hidden_size;
  int const offset = idx % hidden_size;
  // request id in batch config
  int requext_idx_in_batch = -1;
  int cnt_1 = 0;
  while (cnt_1 < request_compact_idx + 1) {
    requext_idx_in_batch++;
    if (request_available[requext_idx_in_batch]) {
      cnt_1++;
    }
  }
  // get the starting index of kv page
  // page attention: WARNING: this implicitly assume that the kv page is stored in the same order as the available requests
  int start = kv_indptr[requext_idx_in_batch];
  int end = kv_indptr[requext_idx_in_batch + 1] - 1;
  for (int i = 0; i < num_committed_tokens; i++) {
    if (committedTokenInfos[i].request_index == requext_idx_in_batch) {
      int const index_in_kv_cache = committedTokenInfos[i].index_in_kv_cache;
      if (index_in_kv_cache == -1) {
        continue;
      }

      // int const req_id = committedTokenInfos[i].request_index;
      int const tok_id = committedTokenInfos[i].token_depth;
      int const page_to_idx = kv_page_indices[start + tok_id / kPagesize];
      assert(start + (tok_id / kPagesize) <= end);
      int const page_from_idx = kv_page_indices[start + index_in_kv_cache / kPagesize];
      assert(start + (index_in_kv_cache / kPagesize) <= end);

      // page attention: since we cannot store temporary tokens in the cache, we need to figure out another way
      // WARNING: we assume that index_in_kv_cache is flattened index in gpu memory
      // index_in_kv_cache is actually the flattened index 
      size_t from_k_idx = get_k_entry_offset(index_in_kv_cache, page_from_idx, hidden_size),
             from_v_idx = get_v_entry_offset(index_in_kv_cache, page_from_idx, hidden_size);

      // page attention: copy the token to the new position
      size_t to_k_idx =get_k_entry_offset(tok_id, page_to_idx, hidden_size),
             to_v_idx =get_v_entry_offset(tok_id, page_to_idx, hidden_size);
      assert(to_k_idx <= from_k_idx);

      kCache_ptr[to_k_idx + offset] = kCache_ptr[from_k_idx + offset];
      kCache_ptr[to_v_idx + offset] = kCache_ptr[from_v_idx + offset];
    }
  }
}

void commit_tokens(TreeIncMultiHeadSelfAttentionMeta const *m,
                   BatchConfig const *bc,
                   cudaStream_t stream) {
  //   cudaEvent_t t_start, t_end;
  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  int num_tokens_to_commit = bc->num_tokens_to_commit;
  int const max_num_pages =
      (BatchConfig::max_sequence_length() +
       BatchConfig::max_spec_tree_token_num() + kPagesize - 1) /
      kPagesize;
  int const num_requests = bc->num_active_requests();
  int parallelism = m->hidden_size * num_requests;
  commit_tokens_kernel<<<GET_BLOCKS(parallelism),
                         min(CUDA_NUM_THREADS, parallelism),
                         0,
                         stream>>>(static_cast<half *>(m->keyCache),
                                   m->handle.tree_verify_attention_metadata->kv_indptr,
                                   m->handle.tree_verify_attention_metadata->kv_indices,
                                   m->committed_token_infos,
                                   m->request_available,
                                   num_requests,
                                   m->hidden_size,
                                   num_tokens_to_commit,
                                   max_num_pages);
  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   float elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   printf("Commit token time: %.2f ms\n", elapsed);
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
}

__global__ void
    update_custom_mask_kernel(float *custom_mask,
                              BatchConfig::BitMask *causalMask,
                              BatchConfig::PerRequestInfo *request_infos,
                              bool *request_available,
                              int const num_requests,
                              int const max_q_length,
                              int const max_kv_length,
                              float mask_value) {
  // get thread idx in [0, num_requests * max_q_length)
  int const idx = blockIdx.x * blockDim.x + threadIdx.x;
  // get (request_idx, q_idx) from thread idx
  int const request_idx = idx / max_q_length / max_kv_length;
  int const q_idx = (idx % (max_q_length * max_kv_length)) / max_kv_length;
  int const kv_idx = idx % max_kv_length;

  // request id in batch config
  int requext_idx_in_batch = -1;
  int cnt_1 = 0, mask_offset = 0, mask_lens = 0;
  while (cnt_1 < request_idx + 1) {
    requext_idx_in_batch++;
    if (request_available[requext_idx_in_batch]) {
      cnt_1++;
      mask_offset = mask_lens;
      int q_len = request_infos[requext_idx_in_batch].num_tokens_in_batch,
          k_len =
              q_len +
              request_infos[requext_idx_in_batch].first_token_index_in_request;
      mask_lens += q_len * k_len;
    }
  }

  int const q_length = request_infos[requext_idx_in_batch].num_tokens_in_batch;
  int const q_start =
      request_infos[requext_idx_in_batch].first_token_index_in_request;
  if (q_idx >= q_length) {
    return;
  }
  if (kv_idx >= q_start + q_length) {
    return;
  }
  assert(q_start + q_length <= max_kv_length);

  float *mask = custom_mask + mask_offset + q_idx * (q_start + q_length);
  if (kv_idx < q_start) {
    mask[kv_idx] = 0.0f;
  } else {
    mask[kv_idx] = test_bit_orig(causalMask[requext_idx_in_batch].bit_mask,
                                 q_idx,
                                 kv_idx - q_start)
                       ? 0.0f
                       : mask_value;
  }
}

void update_custom_mask(TreeIncMultiHeadSelfAttentionMeta const *m,
                        BatchConfig const *bc,
                        cudaStream_t stream) {
  int const num_requests = bc->num_active_requests();
  int const max_q_length = BatchConfig::max_spec_tree_token_num();
  int const max_kv_length = BatchConfig::max_spec_tree_token_num() +
                            BatchConfig::max_sequence_length();
  int parallelism = num_requests * max_q_length * max_kv_length;
  update_custom_mask_kernel<<<GET_BLOCKS(parallelism),
                              min(CUDA_NUM_THREADS, parallelism),
                              0,
                              stream>>>(m->custom_mask,
                                        m->causalMask,
                                        m->request_infos,
                                        m->request_available,
                                        num_requests,
                                        max_q_length,
                                        max_kv_length,
                                        -5e4);
}

template <typename DT>
__global__ void
    update_qkv_cache_kernel(DT *devQKVProjArray,
                            half *qTmp_ptr,
                            half *kCache_ptr,
                            int32_t *kv_indptr,
                            int32_t *kv_page_indices,
                            BatchConfig::PerTokenInfo const *tokenInfos,
                            BatchConfig::PerRequestInfo *request_infos,
                            int const max_num_pages,
                            int hidden_size,
                            int num_new_tokens) {
  int const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int const token_idx = thread_idx / hidden_size;
  int const offset = thread_idx % hidden_size;
  if (token_idx >= num_new_tokens) {
    return;
  }

  int const req_idx = tokenInfos[token_idx].request_index;
  int const token_abs_idx = tokenInfos[token_idx].abs_index_in_request;

  // compute the starting index of kv page
  int start = kv_indptr[req_idx];
  int page_idx = kv_page_indices[start + (token_abs_idx / kPagesize)];
  size_t from_idx = token_idx * QKV_WEIGHT_NUM * hidden_size;
  size_t to_k_idx = get_k_entry_offset(token_abs_idx, page_idx, hidden_size),
         to_v_idx = get_v_entry_offset(token_abs_idx, page_idx, hidden_size);
  // printf("to_k_idx: %lu, to_v_idx: %lu\n", to_k_idx, to_v_idx);

  // key and value cache should be stored interleaved
  kCache_ptr[to_k_idx + offset] =
      static_cast<half>(devQKVProjArray[from_idx + hidden_size + offset]);
  kCache_ptr[to_v_idx + offset] =
      static_cast<half>(devQKVProjArray[from_idx + hidden_size * 2 + offset]);
  qTmp_ptr[token_idx * hidden_size + offset] =
      static_cast<half>(devQKVProjArray[from_idx + offset]);
}

template <typename DT>
void update_qkv_cache(TreeIncMultiHeadSelfAttentionMeta const *m,
                      BatchConfig const *bc,
                      cudaStream_t stream) {
  // update the kv cache, compact the q array
  int num_new_tokens = bc->num_active_tokens();
  int parallelism = m->hidden_size * num_new_tokens;
  int const max_num_pages =
      (BatchConfig::max_sequence_length() +
       BatchConfig::max_spec_tree_token_num() + kPagesize - 1) /
      kPagesize;
  update_qkv_cache_kernel<<<GET_BLOCKS(parallelism),
                            min(CUDA_NUM_THREADS, parallelism),
                            0,
                            stream>>>(static_cast<DT *>(m->devQKVProjArray),
                                      static_cast<half *>(m->queryTmp),
                                      static_cast<half *>(m->keyCache),
                                      m->handle.tree_verify_attention_metadata->kv_indptr,
                                      m->handle.tree_verify_attention_metadata->kv_indices,
                                      m->token_infos,
                                      m->request_infos,
                                      max_num_pages,
                                      m->hidden_size,
                                      num_new_tokens);
}

__global__ void
    prepare_inference_params_kernel(int const num_requests,
                                    BatchConfig::PerRequestInfo *request_infos,
                                    bool *request_available,
                                    uint32_t const max_num_pages,
                                    int32_t *q_indptr,
                                    int32_t *kv_indptr,
                                    int32_t *kv_indices,
                                    int32_t *kv_last_page_len,
                                    int32_t *qk_indptr) {
  int const request_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (request_idx >= num_requests) {
    return;
  }

  // request id in batch config
  int requext_idx_in_batch = -1;
  int cnt_1 = 0, q_lens = 0, qk_len = 0;
  int indices_offset = 0, indices_lens = 0, kv_len = 0;
  while (cnt_1 < request_idx + 1) {
    requext_idx_in_batch++;
    if (request_available[requext_idx_in_batch]) {
      cnt_1++;
      int q_len = request_infos[requext_idx_in_batch].num_tokens_in_batch;
      q_lens += q_len;
      kv_len = request_infos[requext_idx_in_batch].num_tokens_in_batch +
               request_infos[requext_idx_in_batch].first_token_index_in_request;
      qk_len += q_len * kv_len;
      indices_offset = indices_lens;
      indices_lens += (kv_len + kPagesize - 1) / kPagesize;
    }
  }

  if (request_idx == 0) {
    q_indptr[0] = 0;
    kv_indptr[0] = 0;
    qk_indptr[0] = 0;
  }
  __syncthreads();
  q_indptr[request_idx + 1] = q_lens;
  kv_indptr[request_idx + 1] = indices_lens;
  for (int i = indices_offset; i < indices_lens; i++) {
    kv_indices[i] = max_num_pages * requext_idx_in_batch + (i - indices_offset);
  }
  kv_last_page_len[request_idx] = (kv_len - 1) % kPagesize + 1;
  qk_indptr[request_idx + 1] = qk_len;
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
void tree_verify_attention(TreeIncMultiHeadSelfAttentionMeta const *m,
                           BatchConfig const *bc,
                           DT *output_ptr,
                           cudaStream_t stream) {
  //   int device;
  //   checkCUDA(cudaGetDevice(&device));
  //   cudaEvent_t t_start, t_end;
  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  // global constant parameters
  uint32_t const num_q_heads = m->num_q_heads;
  uint32_t const num_kv_heads = m->num_kv_heads;
  uint32_t const group_size = num_q_heads / num_kv_heads;
  uint32_t const head_dim = m->qProjSize;
  uint32_t const max_num_pages =
      (BatchConfig::max_sequence_length() +
       BatchConfig::max_spec_tree_token_num() + kPagesize - 1) /
      kPagesize;
  uint32_t const batch_size = bc->num_active_requests();
  float const sm_scale =
      (*m->qk_prod_scaling) ? 1.0f / sqrt(m->kProjSize) : 1.0f;
  std::vector<int32_t> q_indptr_h{0};

  {
    int parallelism = batch_size;
    prepare_inference_params_kernel<<<GET_BLOCKS(parallelism),
                                      min(CUDA_NUM_THREADS, parallelism),
                                      0,
                                      stream>>>(batch_size,
                                                m->request_infos,
                                                m->request_available,
                                                max_num_pages,
                                                m->q_indptr,
                                                m->kv_indptr,
                                                m->kv_indices,
                                                m->kv_last_page_len,
                                                m->qk_indptr);
    for (int req_idx = 0; req_idx < bc->max_requests_per_batch(); req_idx++) {
      if (bc->request_available[req_idx]) {
        int q_len = bc->requestsInfo[req_idx].num_tokens_in_batch;
        q_indptr_h.push_back(q_indptr_h.back() + q_len);
      }
    }
  }

  half *q = static_cast<half *>(m->queryTmp),
       *kv = static_cast<half *>(m->keyCache),
       *o = static_cast<half *>(m->outputTmp);
  
  static int32_t kv_indices_tmp[BatchConfig::MAX_NUM_REQUESTS * BatchConfig::MAX_NUM_TOKENS];
  static int32_t kv_indptr_tmp[BatchConfig::MAX_NUM_REQUESTS + 1];
  static int32_t kv_last_page_len_tmp[BatchConfig::MAX_NUM_REQUESTS];
  // copy data from device to host
  cudaMemcpy(kv_indices_tmp,
             m->handle.tree_verify_attention_metadata->kv_indices,
             sizeof(int32_t) * BatchConfig::MAX_NUM_REQUESTS * BatchConfig::MAX_NUM_TOKENS,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(kv_indptr_tmp,
              m->handle.tree_verify_attention_metadata->kv_indptr,
              sizeof(int32_t) * (BatchConfig::MAX_NUM_REQUESTS + 1),
              cudaMemcpyDeviceToHost);
  cudaMemcpy(kv_last_page_len_tmp,
              m->handle.tree_verify_attention_metadata->kv_last_page_len,
              sizeof(int32_t) * BatchConfig::MAX_NUM_REQUESTS,
              cudaMemcpyDeviceToHost);
  //print the request information
  // for (int i = 0; i < bc->num_active_requests(); i++) {
  //   printf("request %d: ", i);
  //   for (int j = kv_indptr_tmp[i]; j < kv_indptr_tmp[i + 1]; j++) {
  //     printf("%d ", kv_indices_tmp[j]);
  //   }
  //   printf("\n");
  // }
  // printf("last page length: %d\n", kv_last_page_len_tmp[0]);
  paged_kv_t<PageStorage::kIndices, QKVLayout::kNHD, half, int32_t> paged_kv(
      num_kv_heads,
      kPagesize,
      head_dim,
      batch_size,
      kv,
      m->kv_indices,
      m->kv_indptr,
      m->kv_last_page_len);

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   float elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   if (device == 0) {
  //     printf("    attn prep time: %.4f ms\n", elapsed);
  //   }
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);

  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  BatchPrefillHandler *handler =
      static_cast<BatchPrefillHandler *>(m->batch_prefill_handler);
  handler->SetCUDAStream(stream);
  handler->BeginForward(m->workspace,
                        m->workspace_size,
                        q_indptr_h.data(),
                        batch_size,
                        num_q_heads,
                        num_kv_heads,
                        head_dim);

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   if (device == 0) {
  //     printf("    BeginForward time: %.4f ms\n", elapsed);
  //   }
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);

  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  DISPATCH_GROUPSIZE(
      group_size,
      GROUP_SIZE,
      {DISPATCH_HEADDIM(
          head_dim, HEAD_DIM, {DISPATCH_PAGESIZE(kPagesize, PAGE_SIZE, {
            cudaError_t result;
            if (bc->prompt_phase) {
              result = BatchPrefillWithPagedKVCacheWrapperDispatched<
                  PageStorage::kIndices,
                  QKVLayout::kNHD,
                  PAGE_SIZE,
                  GROUP_SIZE,
                  HEAD_DIM,
                  PosEncodingMode::kNone,
                  false,
                  MaskMode::kCausal,
                  half,
                  half,
                  int32_t>(handler,
                           q,
                           m->q_indptr,
                           /*q_offset=*/nullptr,
                           paged_kv,
                           /*custom_mask=*/nullptr,
                           /*qk_indptr=*/nullptr,
                           o,
                           /*lse=*/nullptr,
                           sm_scale,
                           /*rope_scale=*/1.f,
                           /*rope_theta=*/static_cast<float>(1e4),
                           stream);
            } else {
              result = BatchPrefillWithPagedKVCacheWrapperDispatched<
                  PageStorage::kIndices,
                  QKVLayout::kNHD,
                  PAGE_SIZE,
                  GROUP_SIZE,
                  HEAD_DIM,
                  PosEncodingMode::kNone,
                  false,
                  MaskMode::kCustom,
                  half,
                  half,
                  int32_t>(handler,
                           q,
                           m->q_indptr,
                           /*q_offset=*/nullptr,
                           paged_kv,
                           m->custom_mask,
                           m->qk_indptr,
                           o,
                           /*lse=*/nullptr,
                           sm_scale,
                           /*rope_scale=*/1.f,
                           /*rope_theta=*/static_cast<float>(1e4),
                           stream);
            }
            if (result != cudaSuccess) {
              throw std::runtime_error(
                  "Failed to run "
                  "BatchPrefillWithPagedKVCacheWrapperDispatched" +
                  std::string(cudaGetErrorString(result)));
            }
          })})});

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   if (device == 0) {
  //     printf("    actual attn time: %.4f ms\n", elapsed);
  //   }
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);

  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  {
    int parallelism = m->vProjSize * m->num_q_heads * bc->num_active_tokens();
    produce_output_kernel<<<GET_BLOCKS(parallelism),
                            min(CUDA_NUM_THREADS, parallelism),
                            0,
                            stream>>>(m->outputTmp, output_ptr, parallelism);
  }

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   if (device == 0) {
  //     printf("    produce_output_kernel time: %.4f ms\n", elapsed);
  //   }
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
}

template <typename DT>
void inference_kernel(TreeIncMultiHeadSelfAttentionMeta *m,
                      BatchConfig const *bc,
                      int shard_id,
                      DT const *input_ptr,
                      DT const *weight_ptr,
                      DT *output_ptr,
                      DT const *bias_ptr,
                      cudaStream_t stream) {

  //   int device;
  //   checkCUDA(cudaGetDevice(&device));
  //   cudaEvent_t t_start, t_end;
  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

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
  // std::cout << "tokens to be committed: " << bc->num_tokens_to_commit <<
  // "\n";

  if (!bc->prompt_phase) {
    commit_tokens(m, bc, stream);
  }else{
    // printf("prompt phase in attention\n");
  }

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   float elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
  //   if (device == 0) {
  //     std::cout << "Commit tokens time: " << elapsed << " ms\n";
  //   }

  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  // After commit we update m->num_active_tokens to be the number of active
  // tokens for the current batch
  m->num_active_tokens = bc->num_active_tokens();

  // here because we need postion info in infernece 1
  if (m->offload && m->biasSize > 0) {
    cudaMemcpyAsync(
        m->bias_ptr, bias_ptr, m->biasSize, cudaMemcpyHostToDevice, stream);
    bias_ptr = static_cast<DT *>(m->bias_ptr);
  }
  // Implement kernel to compute KQV for input tokens
  compute_qkv_kernel(m,
                     bc,
                     shard_id,
                     input_ptr,
                     weight_ptr,
                     static_cast<DT *>(m->devQKVProjArray),
                     bias_ptr,
                     stream);

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
  //   if (device == 0) {
  //     std::cout << "Compute qkv time: " << elapsed << " ms\n";
  //   }

  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  // Update gpu-side custom mask referring from CaualMask
  if (!bc->prompt_phase) {
    update_custom_mask(m, bc, stream);
  }

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
  //   if (device == 0) {
  //     std::cout << "Update custom mask time: " << elapsed << " ms\n";
  //   }

  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  // Update key-val cache, compact q array
  update_qkv_cache<DT>(m, bc, stream);

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
  //   if (device == 0) {
  //     std::cout << "Update qkv time: " << elapsed << " ms\n";
  //   }

  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  // Compute attention
  tree_verify_attention<DT>(m, bc, static_cast<DT *>(m->attn_heads), stream);

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
  //   if (device == 0) {
  //     std::cout << "Attn time: " << elapsed << " ms\n";
  //   }

  // Debug output:
  // {
  //   int size = m->hidden_size * bc->num_active_tokens();
  //   float *temp_output = new float[size];
  //   cudaDeviceSynchronize();
  //   cudaMemcpy(
  //       temp_output, m->attn_heads, size * sizeof(float),
  //       cudaMemcpyDeviceToHost);
  //   printf("Output (flashinfer attention) :");
  //   for (int i = 0; i < 1; ++i) {
  //     float temp = 0;
  //     for (int j = 0; j < m->hidden_size; ++j) {
  //       temp += temp_output[i * m->hidden_size + j];
  //     }
  //     printf("%.6f ", temp);
  //   }
  //   printf("\n");

  //   delete[] temp_output;
  // }
  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  int processed_tokens_in_batch = bc->num_active_tokens();

  compute_o_prod_bias(m,
                      bc,
                      shard_id,
                      output_ptr,
                      weight_ptr,
                      bias_ptr,
                      processed_tokens_in_batch,
                      stream);

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
  //   if (device == 0) {
  //     std::cout << "Compute output proj time: " << elapsed << " ms\n";
  //   }
  // {
  //   int size = m->oProjSize;
  //   DT *temp_output = new DT[size];
  //   cudaDeviceSynchronize();
  //   cudaMemcpy(
  //       temp_output, output_ptr + m->oProjSize * (bc->num_active_tokens() -
  //       1), size * sizeof(DT), cudaMemcpyDeviceToHost);
  //   printf("Output :");
  //   for (int i = 0; i < size; ++i) {
  //     printf("%.6f ", static_cast<float>(temp_output[i]));
  //   }
  //   printf("\n");

  //   delete[] temp_output;
  // }
}

} // namespace TreeIncMultiHeadAttention
} // namespace Kernels

/*static*/
void TreeIncMultiHeadSelfAttention::inference_kernel_wrapper(
    TreeIncMultiHeadSelfAttentionMeta *m,
    BatchConfig const *bc,
    int shard_id,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorR const &weight,
    GenericTensorAccessorW const &output,
    GenericTensorAccessorR const &bias) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  bool use_bias = *m->qkv_bias || *m->final_bias;

  //   int device;
  //   checkCUDA(cudaGetDevice(&device));
  //   cudaEvent_t t_start, t_end;
  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

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
    Kernels::TreeIncMultiHeadAttention::inference_kernel<half>(
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
    Kernels::TreeIncMultiHeadAttention::inference_kernel<float>(
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

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   float elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
  //   if (device == 0) {
  //     std::cout << "TreeIncMultiHeadSelfAttention time: " << elapsed << "
  //     ms\n";
  //   }
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

  {
    size_t batch_size = BatchConfig::max_requests_per_batch();
    size_t max_num_pages =
        (BatchConfig::max_spec_tree_token_num() +
         BatchConfig::max_sequence_length() + kPagesize - 1) /
        kPagesize;
    size_t indices_size = std::max(
        (batch_size + 1) * 4 + max_num_pages * batch_size, 1ul * 1024 * 1024);
    size_t custom_mask_size = BatchConfig::max_requests_per_batch() *
                              BatchConfig::max_spec_tree_token_num() *
                              (BatchConfig::max_spec_tree_token_num() +
                               BatchConfig::max_sequence_length());
    workspace_size = 32 * 1024 * 1024; // 32MB

    gpu_mem_allocator.create_legion_instance(
        flashinfer_reserve_inst,
        sizeof(int32_t) * indices_size + sizeof(float) * custom_mask_size +
            workspace_size);

    q_indptr = gpu_mem_allocator.allocate_instance<int32_t>(indices_size);
    kv_indptr = q_indptr + batch_size + 1;
    kv_indices = kv_indptr + batch_size + 1;
    kv_last_page_len = kv_indices + max_num_pages * batch_size;
    qk_indptr = kv_last_page_len + batch_size + 1;
    custom_mask = gpu_mem_allocator.allocate_instance<float>(custom_mask_size);
    workspace = static_cast<void *>(
        gpu_mem_allocator.allocate_instance<char>(workspace_size));
    batch_prefill_handler =
        static_cast<void *>(new flashinfer::BatchPrefillHandler);
  }

  // allocate memory for the seqArray and reserve space
  {
    causalMask = reinterpret_cast<BatchConfig::BitMask *>(
        reinterpret_cast<char *>(handler.batch_config_metadata) +
        sizeof(BatchConfig::tokensInfo) + sizeof(BatchConfig::requestsInfo) +
        sizeof(BatchConfig::request_available));
    committed_token_infos =
        reinterpret_cast<BatchConfig::CommittedTokensInfo *>(
            reinterpret_cast<char *>(handler.batch_config_metadata) +
            sizeof(BatchConfig::tokensInfo) +
            sizeof(BatchConfig::requestsInfo) +
            sizeof(BatchConfig::request_available) +
            sizeof(BatchConfig::causalMask));
  }

  cudaStreamSynchronize(stream);
}

TreeIncMultiHeadSelfAttentionMeta::~TreeIncMultiHeadSelfAttentionMeta(void) {
  if (flashinfer_reserve_inst != Realm::RegionInstance::NO_INST) {
    flashinfer_reserve_inst.destroy();
  }
  delete static_cast<flashinfer::BatchPrefillHandler *>(batch_prefill_handler);
}

}; // namespace FlexFlow

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

#include "flashinfer/decode_attention_decl.cuh"
#include "flashinfer/prefill_attention_decl.cuh"
#include "flexflow/ops/kernels/inc_multihead_self_attention_utils.cuh"
#include "flexflow/request_manager.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

using namespace Legion;

using flashinfer::BatchDecodeHandler;
using flashinfer::BatchPrefillHandler;
using flashinfer::LogitsPostHook;
using flashinfer::paged_kv_t;
using flashinfer::PageStorage;
using flashinfer::PosEncodingMode;
using flashinfer::QKVLayout;

void RequestManager::load_tokens_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);

  // BatchConfig const batch_config = *((BatchConfig *)task->args);
  BatchConfig const *batch_config = BatchConfig::from_future(task->futures[0]);

  BatchConfig::TokenId dram_copy[BatchConfig::MAX_NUM_TOKENS];

  // Extreme long prompts are not supported, only load up to
  // BatchConfig::max_tokens_per_batch() as prompt
  int max_tokens_per_batch =
      std::max(batch_config->get_mode() == TREE_SEARCH_MODE
                   ? BatchConfig::max_tokens_per_ssm_batch()
                   : BatchConfig::max_tokens_per_batch(),
               BatchConfig::max_tokens_per_prefilling_batch());
  if (batch_config->num_tokens > max_tokens_per_batch) {
    printf("Warning: too many tokens in prompt, only load up to %d tokens\n",
           max_tokens_per_batch);
    printf("Got: %d tokens\n", batch_config->num_tokens);
  }

  if (batch_config->num_tokens > 0) {
    for (int i = 0; i < batch_config->num_tokens; i++) {
      dram_copy[i] = batch_config->tokensInfo[i].token_id;
    }
    TokenId *fb_ptr = helperGetTensorPointerWO<TokenId>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    Domain domain = runtime->get_index_space_domain(
        ctx, task->regions[0].region.get_index_space());
    assert(batch_config->num_tokens <= domain.get_volume());
    cudaStream_t stream;
    checkCUDA(get_legion_stream(&stream));
    checkCUDA(cudaMemcpyAsync(fb_ptr,
                              dram_copy,
                              sizeof(TokenId) * batch_config->num_tokens,
                              cudaMemcpyHostToDevice,
                              stream));
  }
}

void prepare_inference_params_kernel_h(BatchConfig const *batch_config,
                                       PageManager *pm,
                                       FFHandler handle,
                                       cudaStream_t stream,
                                       uint32_t const max_num_pages,
                                       int32_t *q_indptr_h,
                                       int32_t *kv_indptr_h,
                                       int32_t *kv_indices_h,
                                       int32_t *kv_last_page_len_h,
                                       int32_t *qk_indptr_h) {
  int batch_size = batch_config->num_active_requests();
  // we just search for the page number for each request
  q_indptr_h[0] = 0;
  kv_indptr_h[0] = 0;
  qk_indptr_h[0] = 0;
  int cnt_1 = 0, q_lens = 0, qk_lens = 0;
  int indices_offset = 0, indices_lens = 0, kv_len = 0;
  for (int req_idx = 0, indptr_idx = 0; req_idx < batch_config->max_requests_per_batch(); req_idx++) {
    if (batch_config->request_available[req_idx]) {
      int q_len = batch_config->requestsInfo[req_idx].num_tokens_in_batch;
      int kv_len = batch_config->requestsInfo[req_idx].num_tokens_in_batch +
                  batch_config->requestsInfo[req_idx].first_token_index_in_request;
      
      q_lens += q_len;
      qk_lens += (q_len * kv_len + 7) / 8;
      indices_offset = indices_lens;
      indices_lens += (kv_len + kPagesize - 1) / kPagesize;
      q_indptr_h[indptr_idx + 1] = q_indptr_h[indptr_idx] + q_len;
      kv_indptr_h[indptr_idx + 1] = batch_config->requestsInfo[req_idx].num_kv_pages + kv_indptr_h[indptr_idx];

      assert(batch_config->requestsInfo[req_idx].num_kv_pages == (kv_len + kPagesize - 1) / kPagesize);
      assert(batch_config->requestsInfo[req_idx].kv_last_page_len <= kPagesize);
      std::vector<int32_t> kv_indices = pm -> get_block_table_indices(batch_config->requestsInfo[req_idx].request_guid);
      printf("request_guid: %d\n", batch_config->requestsInfo[req_idx].request_guid);
      printf("kv_indices.size() = %d, kv_len = %d\n", kv_indices.size(), kv_len);
      printf("kv last page len = %d\n", batch_config->requestsInfo[req_idx].kv_last_page_len);
      printf("num_kv_pages = %d\n", batch_config->requestsInfo[req_idx].num_kv_pages);
      assert(kv_indices.size() == (kv_len + kPagesize - 1) / kPagesize);
      for (int i = indices_offset; i < indices_lens; i++) {
        kv_indices_h[i] = kv_indices[i - indices_offset];
      }
      qk_indptr_h[indptr_idx + 1] = qk_lens;
      kv_last_page_len_h[indptr_idx] = batch_config->requestsInfo[req_idx].kv_last_page_len;
      indptr_idx++;
    }
  }

  // do the copy
  checkCUDA(cudaMemcpyAsync(handle.tree_verify_attention_metadata->kv_indices,
                            kv_indices_h,
                            sizeof(int32_t) * batch_size * max_num_pages,
                            cudaMemcpyHostToDevice,
                            stream));
  checkCUDA(cudaMemcpyAsync(handle.tree_verify_attention_metadata->kv_last_page_len,
                            kv_last_page_len_h,
                            sizeof(int32_t) * batch_size,
                            cudaMemcpyHostToDevice,
                            stream));
  checkCUDA(cudaMemcpyAsync(handle.tree_verify_attention_metadata->q_indptr,
                            q_indptr_h,
                            sizeof(int32_t) * (batch_size + 1),
                            cudaMemcpyHostToDevice,
                            stream));
  checkCUDA(cudaMemcpyAsync(handle.tree_verify_attention_metadata->kv_indptr,
                            kv_indptr_h,
                            sizeof(int32_t) * (batch_size + 1),
                            cudaMemcpyHostToDevice,
                            stream));
  checkCUDA(cudaMemcpyAsync(handle.tree_verify_attention_metadata->qk_indptr,
                            qk_indptr_h,
                            sizeof(int32_t) * (batch_size + 1),
                            cudaMemcpyHostToDevice,
                            stream));
}

// q_indptr: the start offset of q in the batch for each request,
//           the length is `num_requests + 1`: [0, num_q_0, num_q_0 + num_q_1,
//           ..., num_q_0 + num_q_1 + ... + num_q_{num_requests - 1}]
// kv_indptr: the start offset of kv page_indices for each request,
//            the length is `num_requests + 1`.
// kv_indices: the page indices for kv, the length is `num_kv_pages`.
// kv_last_page_len: the cache length in the last page for each request,
//                   the length is `num_requests`.
// qk_indptr: the start offset of custom_mask in the flattened mask for each
//            request, the length is `num_requests + 1`. It can be calculated as
//            accumulative `ceil(qk_len / 8)`.
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
  int cnt_1 = 0, q_lens = 0, qk_lens = 0;
  int indices_offset = 0, indices_lens = 0, kv_len = 0;
  while (cnt_1 < request_idx + 1) {
    requext_idx_in_batch++;
    if (request_available[requext_idx_in_batch]) {
      cnt_1++;
      int q_len = request_infos[requext_idx_in_batch].num_tokens_in_batch;
      q_lens += q_len;
      kv_len = request_infos[requext_idx_in_batch].num_tokens_in_batch +
               request_infos[requext_idx_in_batch].first_token_index_in_request;
      qk_lens += (q_len * kv_len + 7) / 8;
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
  qk_indptr[request_idx + 1] = qk_lens;
}

#define test_bit_orig(bit_mask, idx, pos)                                      \
  (((bit_mask)[idx].bits[(pos) / 64] & (1ULL << ((pos) % 64))) != 0)

// Passing the CPU-side causalMask, then output the bit-packed custom_mask for
// attention forward.
// Layout of causalMask: [num_requests][tree_size][tree_size]
// Layout of custom_mask: [num_requests][q_length][kv_length] (bit-packed)
// Note that for spec-decoding, q_length == last_layer_length != tree_size
__global__ void
    update_custom_mask_kernel(uint8_t *custom_mask,
                              int32_t const *qk_indptr,
                              BatchConfig::BitMask *causalMask,
                              BatchConfig::PerRequestInfo *request_infos,
                              bool *request_available,
                              uint32_t const num_requests) {
  int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int request_idx = 0;
  while (request_idx < num_requests) {
    if (qk_indptr[request_idx + 1] > byte_idx) {
      break;
    }
    request_idx++;
  }

  if (request_idx >= num_requests) {
    return;
  }
  byte_idx -= qk_indptr[request_idx];

  // request id in batch config
  int requext_idx_in_batch = -1, cnt_1 = 0;
  while (cnt_1 < request_idx + 1) {
    requext_idx_in_batch++;
    if (request_available[requext_idx_in_batch]) {
      cnt_1++;
    }
  }

  BatchConfig::BitMask &causal_mask = causalMask[requext_idx_in_batch];

  int const q_length = request_infos[requext_idx_in_batch].num_tokens_in_batch,
            q_start = request_infos[requext_idx_in_batch]
                          .first_token_index_in_request -
                      causal_mask.non_tree_cache_size,
            non_tree_cache_size = causal_mask.non_tree_cache_size;

  uint8_t packed_bits = 0;
  for (int bit_idx = 0; bit_idx < 8; bit_idx++) {
    int const bit_offset = byte_idx * 8 + bit_idx,
              q_idx = bit_offset / (non_tree_cache_size + q_start + q_length),
              kv_idx = bit_offset % (non_tree_cache_size + q_start + q_length);
    if (kv_idx < non_tree_cache_size || q_idx >= q_length) {
      packed_bits |= 1 << bit_idx;
    } else {
      if (test_bit_orig(causal_mask.bit_mask,
                        q_start + q_idx,
                        kv_idx - non_tree_cache_size)) {
        packed_bits |= 1 << bit_idx;
      }
    }
  }
  custom_mask[qk_indptr[request_idx] + byte_idx] = packed_bits;
}

void RequestManager::load_batch_config_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  assert(regions.size() == 0);
  assert(task->regions.size() == 0);
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  // BatchConfig const batch_config = *((BatchConfig *)task->args);
  BatchConfig const *batch_config = BatchConfig::from_future(task->futures[0]);

  // copy meta data to workSpace
  FFHandler handle = *((FFHandler const *)task->local_args);
  size_t total_copy_size = 0;
  if (batch_config->num_tokens > 0) {
    // The tokensInfo is compact
    checkCUDA(cudaMemcpyAsync(handle.batch_config_metadata,
                              &(batch_config->tokensInfo),
                              batch_config->num_tokens *
                                  sizeof(BatchConfig::PerTokenInfo),
                              cudaMemcpyHostToDevice,
                              stream));
  }
  total_copy_size += sizeof(BatchConfig::tokensInfo);

  for (int request_idx = 0; request_idx < BatchConfig::max_requests_per_batch();
       request_idx++) {
    if (batch_config->request_available[request_idx]) {
      checkCUDA(cudaMemcpyAsync(
          static_cast<char *>(handle.batch_config_metadata) + total_copy_size +
              request_idx * sizeof(BatchConfig::PerRequestInfo),
          &(batch_config->requestsInfo[request_idx]),
          sizeof(BatchConfig::PerRequestInfo),
          cudaMemcpyHostToDevice,
          stream));
    }
  }
  total_copy_size += sizeof(BatchConfig::requestsInfo);

  checkCUDA(cudaMemcpyAsync(static_cast<char *>(handle.batch_config_metadata) +
                                total_copy_size,
                            &(batch_config->request_available),
                            sizeof(BatchConfig::request_available),
                            cudaMemcpyHostToDevice,
                            stream));
  total_copy_size += sizeof(BatchConfig::request_available);

  for (int request_idx = 0; request_idx < BatchConfig::max_requests_per_batch();
       request_idx++) {
    if (batch_config->request_available[request_idx]) {
      checkCUDA(cudaMemcpyAsync(
          static_cast<char *>(handle.batch_config_metadata) + total_copy_size +
              request_idx * sizeof(BatchConfig::BitMask),
          &(batch_config->causalMask[request_idx]),
          sizeof(BatchConfig::BitMask),
          cudaMemcpyHostToDevice,
          stream));
    }
  }
  total_copy_size += sizeof(BatchConfig::causalMask);

  checkCUDA(cudaMemcpyAsync(static_cast<char *>(handle.batch_config_metadata) +
                                total_copy_size,
                            &(batch_config->streamingCacheInfo),
                            sizeof(BatchConfig::streamingCacheInfo),
                            cudaMemcpyHostToDevice,
                            stream));
  total_copy_size += sizeof(BatchConfig::streamingCacheInfo);

  if (batch_config->num_tokens_to_commit > 0) {
    checkCUDA(cudaMemcpyAsync(
        static_cast<char *>(handle.batch_config_metadata) + total_copy_size,
        &(batch_config->committed_tokens),
        batch_config->num_tokens_to_commit *
            sizeof(BatchConfig::CommittedTokensInfo),
        cudaMemcpyHostToDevice,
        stream));
  }
  total_copy_size += sizeof(BatchConfig::committed_tokens);

  checkCUDA(cudaMemcpyAsync(static_cast<char *>(handle.batch_config_metadata) +
                                total_copy_size,
                            &(batch_config->num_tokens_to_commit),
                            sizeof(int),
                            cudaMemcpyHostToDevice,
                            stream));
  total_copy_size += sizeof(int);

  // load attention metadata
  if (batch_config->get_mode() == INC_DECODING_MODE) {
    if (handle.incr_attention_metadata->enabled()) {
      // calculate the attention meta data
      {
        BatchConfig::PerRequestInfo *request_infos =
            reinterpret_cast<BatchConfig::PerRequestInfo *>(
                static_cast<char *>(handle.batch_config_metadata) +
                sizeof(BatchConfig::tokensInfo));
        bool *request_available = reinterpret_cast<bool *>(
            static_cast<char *>(handle.batch_config_metadata) +
            sizeof(BatchConfig::tokensInfo) +
            sizeof(BatchConfig::requestsInfo));
        int batch_size = batch_config->num_active_requests();
        uint32_t const max_num_pages =
            round_up_pages(BatchConfig::max_sequence_length() +
                           BatchConfig::max_spec_tree_token_num());

        int parallelism = batch_size;
        prepare_inference_params_kernel<<<GET_BLOCKS(parallelism),
                                          min(CUDA_NUM_THREADS, parallelism),
                                          0,
                                          stream>>>(
            batch_size,
            request_infos,
            request_available,
            max_num_pages,
            handle.incr_attention_metadata->q_indptr,
            handle.incr_attention_metadata->kv_indptr,
            handle.incr_attention_metadata->kv_indices,
            handle.incr_attention_metadata->kv_last_page_len,
            handle.incr_attention_metadata->qk_indptr);
      }

      // prepare attention forward handler
      {
        int batch_size = batch_config->num_active_requests();
        static int32_t q_indptr_h[BatchConfig::MAX_NUM_REQUESTS + 1],
            kv_indptr_h[BatchConfig::MAX_NUM_REQUESTS + 1],
            kv_last_page_len_h[BatchConfig::MAX_NUM_REQUESTS];
        q_indptr_h[0] = 0;
        kv_indptr_h[0] = 0;
        for (int req_idx = 0, indptr_idx = 0;
             req_idx < batch_config->max_requests_per_batch();
             req_idx++) {
          if (batch_config->request_available[req_idx]) {
            int q_len = batch_config->requestsInfo[req_idx].num_tokens_in_batch;
            int kv_len =
                batch_config->requestsInfo[req_idx].num_tokens_in_batch +
                batch_config->requestsInfo[req_idx]
                    .first_token_index_in_request;
            q_indptr_h[indptr_idx + 1] = q_indptr_h[indptr_idx] + q_len;
            kv_indptr_h[indptr_idx + 1] =
                kv_indptr_h[indptr_idx] + round_up_pages(kv_len);
            kv_last_page_len_h[indptr_idx] = (kv_len - 1) % kPagesize + 1;
            indptr_idx++;
          }
        }

        if (!batch_config->prompt_phase) {
          BatchDecodeHandler *handler = nullptr;
          if (handle.incr_attention_metadata->decode_handler_collections.count(
                  batch_size) == 0) {
            handle.incr_attention_metadata
                ->decode_handler_collections[batch_size] = static_cast<void *>(
                new flashinfer::BatchDecodeHandler(true, batch_size));
          }
          handler = static_cast<BatchDecodeHandler *>(
              handle.incr_attention_metadata
                  ->decode_handler_collections[batch_size]);

          handler->SetCUDAStream(stream);
          DISPATCH_HEADDIM(
              handle.incr_attention_metadata->head_dim(), HEAD_DIM, {
                handler->BeginForwardDispatched<HEAD_DIM,
                                                PageStorage::kIndices,
                                                LogitsPostHook::kNone,
                                                PosEncodingMode::kNone,
                                                half,
                                                half,
                                                half,
                                                int32_t>(
                    static_cast<void *>(
                        handle.incr_attention_metadata->float_workspace),
                    handle.incr_attention_metadata->float_workspace_size,
                    static_cast<void *>(
                        handle.incr_attention_metadata->int_workspace),
                    handle.incr_attention_metadata->int_workspace_size,
                    static_cast<int32_t *>(kv_indptr_h),
                    static_cast<int32_t *>(kv_last_page_len_h),
                    batch_size,
                    handle.incr_attention_metadata->num_q_heads(),
                    handle.incr_attention_metadata->num_kv_heads(),
                    kPagesize);
              });
        } else {
          BatchPrefillHandler *handler = nullptr;
          if (handle.incr_attention_metadata->prompt_handler_collections.count(
                  batch_size) == 0) {
            handle.incr_attention_metadata
                ->prompt_handler_collections[batch_size] =
                static_cast<void *>(new flashinfer::BatchPrefillHandler(true));
          }
          handler = static_cast<BatchPrefillHandler *>(
              handle.incr_attention_metadata
                  ->prompt_handler_collections[batch_size]);

          handler->SetCUDAStream(stream);
          handler->BeginForward<half, int32_t>(
              static_cast<void *>(
                  handle.incr_attention_metadata->float_workspace),
              handle.incr_attention_metadata->float_workspace_size,
              static_cast<void *>(
                  handle.incr_attention_metadata->int_workspace),
              handle.incr_attention_metadata->int_workspace_size,
              static_cast<int32_t *>(q_indptr_h),
              static_cast<int32_t *>(kv_indptr_h),
              batch_size,
              handle.incr_attention_metadata->num_q_heads(),
              handle.incr_attention_metadata->num_kv_heads(),
              handle.incr_attention_metadata->head_dim(),
              kPagesize);
        }
      }
    }
  } else if (batch_config->get_mode() == TREE_SEARCH_MODE) {
    if (handle.tree_search_attention_metadata->enabled()) {
      // calculate the attention meta data
      {
        BatchConfig::PerRequestInfo *request_infos =
            reinterpret_cast<BatchConfig::PerRequestInfo *>(
                static_cast<char *>(handle.batch_config_metadata) +
                sizeof(BatchConfig::tokensInfo));
        bool *request_available = reinterpret_cast<bool *>(
            static_cast<char *>(handle.batch_config_metadata) +
            sizeof(BatchConfig::tokensInfo) +
            sizeof(BatchConfig::requestsInfo));
        BatchConfig::BitMask *causalMask =
            reinterpret_cast<BatchConfig::BitMask *>(
                static_cast<char *>(handle.batch_config_metadata) +
                sizeof(BatchConfig::tokensInfo) +
                sizeof(BatchConfig::requestsInfo) +
                sizeof(BatchConfig::request_available));
        int batch_size = batch_config->num_active_requests();
        uint32_t const max_num_pages =
            round_up_pages(BatchConfig::max_sequence_length() +
                           BatchConfig::max_spec_tree_token_num());

        int parallelism = batch_size;
        prepare_inference_params_kernel<<<GET_BLOCKS(parallelism),
                                          min(CUDA_NUM_THREADS, parallelism),
                                          0,
                                          stream>>>(
            batch_size,
            request_infos,
            request_available,
            max_num_pages,
            handle.tree_search_attention_metadata->q_indptr,
            handle.tree_search_attention_metadata->kv_indptr,
            handle.tree_search_attention_metadata->kv_indices,
            handle.tree_search_attention_metadata->kv_last_page_len,
            handle.tree_search_attention_metadata->qk_indptr);

        // Update gpu-side custom mask referring from CaualMask
        if (!batch_config->prompt_phase) {
          int parallelism = 0;
          for (int req_idx = 0;
               req_idx < batch_config->max_requests_per_batch();
               req_idx++) {
            if (batch_config->request_available[req_idx]) {
              int q_len =
                  batch_config->requestsInfo[req_idx].num_tokens_in_batch;
              int kv_len =
                  batch_config->requestsInfo[req_idx].num_tokens_in_batch +
                  batch_config->requestsInfo[req_idx]
                      .first_token_index_in_request;
              parallelism += (q_len * kv_len + 7) / 8;
            }
          }
          update_custom_mask_kernel<<<GET_BLOCKS(parallelism),
                                      min(CUDA_NUM_THREADS, parallelism),
                                      0,
                                      stream>>>(
              handle.tree_search_attention_metadata->custom_mask,
              handle.tree_search_attention_metadata->qk_indptr,
              causalMask,
              request_infos,
              request_available,
              batch_size);
        }
      }

      // prepare attention forward handler
      {
        int batch_size = batch_config->num_active_requests();
        BatchPrefillHandler *handler = nullptr;

        if (!batch_config->prompt_phase) {
          if (handle.tree_search_attention_metadata->decode_handler_collections
                  .count(batch_size) == 0) {
            handle.tree_search_attention_metadata
                ->decode_handler_collections[batch_size] =
                static_cast<void *>(new flashinfer::BatchPrefillHandler(true));
          }
          handler = static_cast<BatchPrefillHandler *>(
              handle.tree_search_attention_metadata
                  ->decode_handler_collections[batch_size]);
        } else {
          if (handle.tree_search_attention_metadata->prompt_handler_collections
                  .count(batch_size) == 0) {
            handle.tree_search_attention_metadata
                ->prompt_handler_collections[batch_size] =
                static_cast<void *>(new flashinfer::BatchPrefillHandler(true));
          }
          handler = static_cast<BatchPrefillHandler *>(
              handle.tree_search_attention_metadata
                  ->prompt_handler_collections[batch_size]);
        }

        static int32_t q_indptr_h[BatchConfig::MAX_NUM_REQUESTS + 1],
            kv_indptr_h[BatchConfig::MAX_NUM_REQUESTS + 1];
        q_indptr_h[0] = 0;
        kv_indptr_h[0] = 0;
        for (int req_idx = 0, indptr_idx = 0;
             req_idx < batch_config->max_requests_per_batch();
             req_idx++) {
          if (batch_config->request_available[req_idx]) {
            int q_len = batch_config->requestsInfo[req_idx].num_tokens_in_batch;
            int kv_len =
                batch_config->requestsInfo[req_idx].num_tokens_in_batch +
                batch_config->requestsInfo[req_idx]
                    .first_token_index_in_request;
            q_indptr_h[indptr_idx + 1] = q_indptr_h[indptr_idx] + q_len;
            kv_indptr_h[indptr_idx + 1] =
                kv_indptr_h[indptr_idx] + round_up_pages(kv_len);
            indptr_idx++;
          }
        }

        handler->SetCUDAStream(stream);
        handler->BeginForward<half, int32_t>(
            static_cast<void *>(
                handle.tree_search_attention_metadata->float_workspace),
            handle.tree_search_attention_metadata->float_workspace_size,
            static_cast<void *>(
                handle.tree_search_attention_metadata->int_workspace),
            handle.tree_search_attention_metadata->int_workspace_size,
            static_cast<int32_t *>(q_indptr_h),
            static_cast<int32_t *>(kv_indptr_h),
            batch_size,
            handle.tree_search_attention_metadata->num_q_heads(),
            handle.tree_search_attention_metadata->num_kv_heads(),
            handle.tree_search_attention_metadata->head_dim(),
            kPagesize);
      }
    }
  } else if (batch_config->get_mode() == TREE_VERIFY_MODE) {
    PageManager *pm = PageManager::get_page_manager();
    // hardcode request 
    // printf("request has allocated %d pages\n", pm -> get_block_table_indices(1000000).size());
    static int32_t q_indptr_h[BatchConfig::MAX_NUM_REQUESTS + 1], kv_indptr_h[BatchConfig::MAX_NUM_REQUESTS + 1];
    static int32_t kv_indices_h[BatchConfig::MAX_NUM_REQUESTS * BatchConfig::MAX_NUM_TOKENS];
    static int32_t qk_indptr_h[BatchConfig::MAX_NUM_REQUESTS + 1];
    static int32_t kv_last_page_len_h[BatchConfig::MAX_NUM_REQUESTS];

    if (handle.tree_verify_attention_metadata->enabled()) {
      // calculate the attention meta data
      {
        BatchConfig::PerRequestInfo *request_infos =
            reinterpret_cast<BatchConfig::PerRequestInfo *>(
                static_cast<char *>(handle.batch_config_metadata) +
                sizeof(BatchConfig::tokensInfo));
        bool *request_available = reinterpret_cast<bool *>(
            static_cast<char *>(handle.batch_config_metadata) +
            sizeof(BatchConfig::tokensInfo) +
            sizeof(BatchConfig::requestsInfo));
        BatchConfig::BitMask *causalMask =
            reinterpret_cast<BatchConfig::BitMask *>(
                static_cast<char *>(handle.batch_config_metadata) +
                sizeof(BatchConfig::tokensInfo) +
                sizeof(BatchConfig::requestsInfo) +
                sizeof(BatchConfig::request_available));
        int batch_size = batch_config->num_active_requests();
        uint32_t const max_num_pages =
            round_up_pages(BatchConfig::max_sequence_length() +
                           BatchConfig::max_spec_tree_token_num());

        int parallelism = batch_size;
        // prepare_inference_params_kernel<<<GET_BLOCKS(parallelism),
        //                                   min(CUDA_NUM_THREADS, parallelism),
        //                                   0,
        //                                   stream>>>(
        //     batch_size,
        //     request_infos,
        //     request_available,
        //     max_num_pages,
        //     handle.tree_verify_attention_metadata->q_indptr,
        //     handle.tree_verify_attention_metadata->kv_indptr,
        //     handle.tree_verify_attention_metadata->kv_indices,
        //     handle.tree_verify_attention_metadata->kv_last_page_len,
        //     handle.tree_verify_attention_metadata->qk_indptr);
        prepare_inference_params_kernel_h(batch_config,
                                          pm,
                                          handle,
                                          stream,
                                          max_num_pages,
                                          q_indptr_h,
                                          kv_indptr_h,
                                          kv_indices_h,
                                          kv_last_page_len_h,
                                          qk_indptr_h);

        // Update gpu-side custom mask referring from CaualMask
        if (!batch_config->prompt_phase) {
          int parallelism = 0;
          for (int req_idx = 0;
               req_idx < batch_config->max_requests_per_batch();
               req_idx++) {
            if (batch_config->request_available[req_idx]) {
              int q_len =
                  batch_config->requestsInfo[req_idx].num_tokens_in_batch;
              int kv_len =
                  batch_config->requestsInfo[req_idx].num_tokens_in_batch +
                  batch_config->requestsInfo[req_idx]
                      .first_token_index_in_request;
              parallelism += (q_len * kv_len + 7) / 8;
            }
          }
          update_custom_mask_kernel<<<GET_BLOCKS(parallelism),
                                      min(CUDA_NUM_THREADS, parallelism),
                                      0,
                                      stream>>>(
              handle.tree_verify_attention_metadata->custom_mask,
              handle.tree_verify_attention_metadata->qk_indptr,
              causalMask,
              request_infos,
              request_available,
              batch_size);
        }
      }

      // prepare attention forward handler
      {
        int batch_size = batch_config->num_active_requests();
        BatchPrefillHandler *handler = nullptr;

        if (!batch_config->prompt_phase) {
          if (handle.tree_verify_attention_metadata->decode_handler_collections
                  .count(batch_size) == 0) {
            handle.tree_verify_attention_metadata
                ->decode_handler_collections[batch_size] =
                static_cast<void *>(new flashinfer::BatchPrefillHandler(true));
          }
          handler = static_cast<BatchPrefillHandler *>(
              handle.tree_verify_attention_metadata
                  ->decode_handler_collections[batch_size]);
        } else {
          if (handle.tree_verify_attention_metadata->prompt_handler_collections
                  .count(batch_size) == 0) {
            handle.tree_verify_attention_metadata
                ->prompt_handler_collections[batch_size] =
                static_cast<void *>(new flashinfer::BatchPrefillHandler(true));
          }
          handler = static_cast<BatchPrefillHandler *>(
              handle.tree_verify_attention_metadata
                  ->prompt_handler_collections[batch_size]);
        }

        // static int32_t q_indptr_h[BatchConfig::MAX_NUM_REQUESTS + 1],
        //     kv_indptr_h[BatchConfig::MAX_NUM_REQUESTS + 1];
        // q_indptr_h[0] = 0;
        // kv_indptr_h[0] = 0;
        // for (int req_idx = 0, indptr_idx = 0;
        //      req_idx < batch_config->max_requests_per_batch();
        //      req_idx++) {
        //   if (batch_config->request_available[req_idx]) {
        //     int q_len = batch_config->requestsInfo[req_idx].num_tokens_in_batch;
        //     int kv_len =
        //         batch_config->requestsInfo[req_idx].num_tokens_in_batch +
        //         batch_config->requestsInfo[req_idx]
        //             .first_token_index_in_request;
        //     q_indptr_h[indptr_idx + 1] = q_indptr_h[indptr_idx] + q_len;
        //     kv_indptr_h[indptr_idx + 1] =
        //         kv_indptr_h[indptr_idx] + round_up_pages(kv_len);
        //     indptr_idx++;
        //   }
        // }

        handler->SetCUDAStream(stream);
        handler->BeginForward<half, int32_t>(
            static_cast<void *>(
                handle.tree_verify_attention_metadata->float_workspace),
            handle.tree_verify_attention_metadata->float_workspace_size,
            static_cast<void *>(
                handle.tree_verify_attention_metadata->int_workspace),
            handle.tree_verify_attention_metadata->int_workspace_size,
            static_cast<int32_t *>(q_indptr_h),
            static_cast<int32_t *>(kv_indptr_h),
            batch_size,
            handle.tree_verify_attention_metadata->num_q_heads(),
            handle.tree_verify_attention_metadata->num_kv_heads(),
            handle.tree_verify_attention_metadata->head_dim(),
            kPagesize);

            cudaError_t syncErr = cudaDeviceSynchronize();
            if (syncErr != cudaSuccess) {
              printf("Kernel execution error: %s\n", cudaGetErrorString(syncErr));
              assert(false);
            }
      }
    }
  }

  // add a size check
  assert(total_copy_size <= handle.batch_config_metadata_size);
}

void RequestManager::load_positions_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);

  // BatchConfig const batch_config = *((BatchConfig *)task->args);
  BatchConfig const *batch_config = BatchConfig::from_future(task->futures[0]);

  int const offset = *((int const *)task->args);
  int *pos_ptr = helperGetTensorPointerWO<int>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  int dram_copy[BatchConfig::MAX_NUM_TOKENS];

  for (int i = 0; i < batch_config->num_tokens; i++) {
    dram_copy[i] = batch_config->tokensInfo[i].abs_index_in_request + offset;
  }

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDA(cudaMemcpyAsync(pos_ptr,
                            dram_copy,
                            sizeof(int) * batch_config->num_tokens,
                            cudaMemcpyHostToDevice,
                            stream));
}

}; // namespace FlexFlow

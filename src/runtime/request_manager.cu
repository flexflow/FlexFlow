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

#include "flexflow/request_manager.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

using namespace Legion;

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
  if (batch_config->num_tokens > BatchConfig::max_tokens_per_batch() &&
      batch_config->get_mode() == INC_DECODING_MODE) {
    printf("Warning: too many tokens in prompt, only load up to %d tokens\n",
           BatchConfig::max_tokens_per_batch());
    printf("Got: %d tokens\n", batch_config->num_tokens);
  } else if (batch_config->num_tokens >
             BatchConfig::max_verify_tokens_per_batch()) {
    printf("Warning: Speculative decoding. too many tokens in prompt, only "
           "load up to %d tokens\n",
           BatchConfig::max_verify_tokens_per_batch());
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

  // load speculative metadata
  if (batch_config->get_mode() == TREE_SEARCH_MODE) {
    for (int request_idx = 0;
         request_idx < BatchConfig::max_requests_per_batch();
         request_idx++) {
      if (batch_config->request_available[request_idx]) {
        checkCUDA(cudaMemcpyAsync(
            static_cast<char *>(handle.batch_config_metadata) +
                total_copy_size + request_idx * sizeof(BatchConfig::BitMask),
            &(batch_config->causalMask[request_idx]),
            sizeof(BatchConfig::BitMask),
            cudaMemcpyHostToDevice,
            stream));
      }
    }
    total_copy_size += sizeof(BatchConfig::causalMask);
  } else if (batch_config->get_mode() == TREE_VERIFY_MODE) {
    for (int request_idx = 0;
         request_idx < BatchConfig::max_requests_per_batch();
         request_idx++) {
      if (batch_config->request_available[request_idx]) {
        checkCUDA(cudaMemcpyAsync(
            static_cast<char *>(handle.batch_config_metadata) +
                total_copy_size + request_idx * sizeof(BatchConfig::BitMask),
            &(batch_config->causalMask[request_idx]),
            sizeof(BatchConfig::BitMask),
            cudaMemcpyHostToDevice,
            stream));
      }
      total_copy_size += sizeof(BatchConfig::committed_tokens);

      // calculate the attention meta data
      {
        BatchConfig::PerRequestInfo *request_infos = reinterpret_cast<BatchConfig::PerRequestInfo *>(
          static_cast<char *>(handle.batch_config_metadata) +
          sizeof(BatchConfig::tokensInfo));
        bool *request_available = reinterpret_cast<bool *>(
          static_cast<char *>(handle.batch_config_metadata) +
          sizeof(BatchConfig::tokensInfo) +
          sizeof(BatchConfig::requestsInfo));
        BatchConfig::BitMask *causalMask = reinterpret_cast<BatchConfig::BitMask *>(
          static_cast<char *>(handle.batch_config_metadata) +
          sizeof(BatchConfig::tokensInfo) +
          sizeof(BatchConfig::requestsInfo) +
          sizeof(BatchConfig::request_available));
        int batch_size = batch_config->num_active_requests();
        uint32_t const max_num_pages = (BatchConfig::max_sequence_length() +
          BatchConfig::max_spec_tree_token_num() + kPagesize - 1) / kPagesize;

        int parallelism = batch_size;

        // Update gpu-side custom mask referring from CaualMask
        if (!batch_config->prompt_phase) {
          int parallelism = 0;
          for (int req_idx = 0; req_idx < batch_config->max_requests_per_batch(); req_idx++) {
            if (batch_config->request_available[req_idx]) {
              int q_len = batch_config->requestsInfo[req_idx].num_tokens_in_batch;
              int kv_len = batch_config->requestsInfo[req_idx].num_tokens_in_batch +
                          batch_config->requestsInfo[req_idx].first_token_index_in_request;
              parallelism += (q_len * kv_len + 7) / 8;
            }
          }
          update_custom_mask_kernel<<<GET_BLOCKS(parallelism),
                                      min(CUDA_NUM_THREADS, parallelism),
                                      0,
                                      stream>>>(handle.tree_verify_attention_metadata->custom_mask,
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
        uint32_t const max_num_pages = (BatchConfig::max_sequence_length() +
          BatchConfig::max_spec_tree_token_num() + kPagesize - 1) / kPagesize;
        BatchPrefillHandler *handler = nullptr;

        if (!batch_config->prompt_phase) {
          if (handle.tree_verify_attention_metadata->decode_handler_collections.count(batch_size) == 0) {
            handle.tree_verify_attention_metadata->decode_handler_collections[batch_size] =
                static_cast<void *>(new flashinfer::BatchPrefillHandler(true));
          }
          handler = static_cast<BatchPrefillHandler *>(
            handle.tree_verify_attention_metadata->decode_handler_collections[batch_size]);
        } else {
          if (handle.tree_verify_attention_metadata->prompt_handler_collections.count(batch_size) == 0) {
            handle.tree_verify_attention_metadata->prompt_handler_collections[batch_size] =
                static_cast<void *>(new flashinfer::BatchPrefillHandler(true));
          }
          handler = static_cast<BatchPrefillHandler *>(
            handle.tree_verify_attention_metadata->prompt_handler_collections[batch_size]);
        }

        if (!batch_config->prompt_phase) {
          if (handle.tree_verify_attention_metadata->decode_handler_collections.count(batch_size) == 0) {
            handle.tree_verify_attention_metadata->decode_handler_collections[batch_size] =
                static_cast<void *>(new flashinfer::BatchPrefillHandler(true));
          }
          handler = static_cast<BatchPrefillHandler *>(
            handle.tree_verify_attention_metadata->decode_handler_collections[batch_size]);
        } else {
          if (handle.tree_verify_attention_metadata->prompt_handler_collections.count(batch_size) == 0) {
            handle.tree_verify_attention_metadata->prompt_handler_collections[batch_size] =
                static_cast<void *>(new flashinfer::BatchPrefillHandler(true));
          }
          handler = static_cast<BatchPrefillHandler *>(
            handle.tree_verify_attention_metadata->prompt_handler_collections[batch_size]);
        }

        static int32_t q_indptr_h[BatchConfig::MAX_NUM_REQUESTS + 1], kv_indptr_h[BatchConfig::MAX_NUM_REQUESTS + 1];
        static int32_t kv_indices_h[BatchConfig::MAX_NUM_REQUESTS * BatchConfig::MAX_NUM_TOKENS];
        static int32_t qk_indptr_h[BatchConfig::MAX_NUM_REQUESTS + 1];
        static int32_t kv_last_page_len_h[BatchConfig::MAX_NUM_REQUESTS];
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
            kv_indptr_h[indptr_idx + 1] = kv_indptr_h[indptr_idx] + (kv_len + kPagesize - 1) / kPagesize;
            for (int i = indices_offset; i < indices_lens; i++) {
              kv_indices_h[i] = max_num_pages * req_idx  + (i - indices_offset);
            }
            qk_indptr_h[indptr_idx + 1] = qk_lens;
            kv_last_page_len_h[indptr_idx] = (kv_len - 1) % kPagesize + 1;
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

        handler->SetCUDAStream(stream);
        handler->BeginForward<half, int32_t>(static_cast<void*>(
                                              static_cast<char*>(handle.tree_verify_attention_metadata->workspace) +
                                              handle.tree_verify_attention_metadata->workspace_block * batch_size),
                                            handle.tree_verify_attention_metadata->workspace_block,
                                            static_cast<int32_t *>(q_indptr_h),
                                            static_cast<int32_t *>(kv_indptr_h),
                                            batch_size,
                                            handle.tree_verify_attention_metadata->num_q_heads(),
                                            handle.tree_verify_attention_metadata->num_kv_heads(),
                                            handle.tree_verify_attention_metadata->head_dim(),
                                            kPagesize);
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

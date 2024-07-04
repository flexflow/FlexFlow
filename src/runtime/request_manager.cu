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
    }
    total_copy_size += sizeof(BatchConfig::causalMask);

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

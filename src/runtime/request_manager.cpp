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
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

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
  // max_tokens_per_batch as prompt
  if (batch_config->num_tokens > BatchConfig::max_tokens_per_batch()) {
    printf("Warning: too many tokens in prompt, only load up to %d tokens\n",
           BatchConfig::max_tokens_per_batch());
    printf("Got: %d tokens\n", batch_config->num_tokens);
  }

  for (int i = 0; i < batch_config->num_tokens; i++) {
    dram_copy[i] = batch_config->tokensInfo[i].token_id;
  }
  TokenId *fb_ptr = helperGetTensorPointerWO<TokenId>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(batch_config->num_tokens <= domain.get_volume());
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDA(hipMemcpyAsync(fb_ptr,
                           dram_copy,
                           sizeof(TokenId) * batch_config->num_tokens,
                           hipMemcpyHostToDevice,
                           stream));
}

void RequestManager::load_batch_config_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  assert(regions.size() == 0);
  assert(task->regions.size() == 0);
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  // BatchConfig const batch_config = *((BatchConfig *)task->args);
  BatchConfig const *batch_config = BatchConfig::from_future(task->futures[0]);

  // copy meta data to workSpace
  FFHandler handle = *((FFHandler const *)task->local_args);
  size_t total_copy_size = 0;
  checkCUDA(hipMemcpyAsync(handle.batch_config_metadata,
                           &(batch_config->tokensInfo),
                           sizeof(BatchConfig::tokensInfo),
                           hipMemcpyHostToDevice,
                           stream));
  total_copy_size += sizeof(BatchConfig::tokensInfo);

  checkCUDA(hipMemcpyAsync(static_cast<char *>(handle.batch_config_metadata) +
                               total_copy_size,
                           &(batch_config->requestsInfo),
                           sizeof(BatchConfig::requestsInfo),
                           hipMemcpyHostToDevice,
                           stream));
  total_copy_size += sizeof(BatchConfig::requestsInfo);

  // load speculative metadata
  if (batch_config->get_mode() == BEAM_SEARCH_MODE) {
    BeamSearchBatchConfig const *beam_batch_config =
        static_cast<BeamSearchBatchConfig const *>(batch_config);

    checkCUDA(hipMemcpyAsync(static_cast<char *>(handle.batch_config_metadata) +
                                 total_copy_size,
                             &(beam_batch_config->beamTokenInfo),
                             sizeof(BeamSearchBatchConfig::beamTokenInfo),
                             hipMemcpyHostToDevice,
                             stream));

    total_copy_size += sizeof(BeamSearchBatchConfig::beamTokenInfo);

    checkCUDA(hipMemcpyAsync(static_cast<char *>(handle.batch_config_metadata) +
                                 total_copy_size,
                             &(beam_batch_config->beamRequestsInfo),
                             sizeof(BeamSearchBatchConfig::beamRequestsInfo),
                             hipMemcpyHostToDevice,
                             stream));
    total_copy_size += sizeof(BeamSearchBatchConfig::beamRequestsInfo);

    checkCUDA(hipMemcpyAsync(static_cast<char *>(handle.batch_config_metadata) +
                                 total_copy_size,
                             &(beam_batch_config->causalMask),
                             sizeof(BatchConfig::causalMask),
                             hipMemcpyHostToDevice,
                             stream));

    total_copy_size += sizeof(BatchConfig::causalMask);
  } else if (batch_config->get_mode() == TREE_VERIFY_MODE) {
    TreeVerifyBatchConfig const *tree_batch_config =
        static_cast<TreeVerifyBatchConfig const *>(batch_config);

    checkCUDA(hipMemcpyAsync(static_cast<char *>(handle.batch_config_metadata) +
                                 total_copy_size,
                             &(tree_batch_config->causalMask),
                             sizeof(BatchConfig::causalMask),
                             hipMemcpyHostToDevice,
                             stream));
    total_copy_size += sizeof(BatchConfig::causalMask);
    checkCUDA(hipMemcpyAsync(static_cast<char *>(handle.batch_config_metadata) +
                                 total_copy_size,
                             &(tree_batch_config->committed_tokens),
                             sizeof(TreeVerifyBatchConfig::committed_tokens),
                             hipMemcpyHostToDevice,
                             stream));
    total_copy_size += sizeof(TreeVerifyBatchConfig::committed_tokens);
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
  BatchConfig const *batch_config = BatchConfig::from_future(task->futures[0]);

  int const offset = *((int const *)task->args);
  int *pos_ptr = helperGetTensorPointerWO<int>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  int dram_copy[BatchConfig::MAX_NUM_TOKENS];

  for (int i = 0; i < batch_config->num_tokens; i++) {
    dram_copy[i] = batch_config->tokensInfo[i].abs_depth_in_request + offset;
  }
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDA(hipMemcpyAsync(pos_ptr,
                           dram_copy,
                           sizeof(int) * batch_config->num_tokens,
                           hipMemcpyHostToDevice,
                           stream));
}

}; // namespace FlexFlow

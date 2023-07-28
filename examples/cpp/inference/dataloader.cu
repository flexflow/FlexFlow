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

#include "dataloader.h"
#include "flexflow/inference.h"
#include "flexflow/request_manager.h"
#include "flexflow/utils/cuda_helper.h"

void DataLoader::load_input(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);

  DataLoaderNextBatchInput const input_struct =
      *((DataLoaderNextBatchInput *)task->args);

  BatchConfig *bc = input_struct.bc;
  BatchConfig::PerRequestInfo *requestInfo = bc->requestsInfo;
  BatchConfig::PerTokenInfo *tokensInfo = bc->tokensInfo;
  std::map<size_t, int> const &prev_batch_preds = input_struct.prev_batch_preds;

  if (bc->num_active_tokens() == 0) {
    return;
  }
  int const *full_input_ptr = helperGetTensorPointerRO<int>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  int *batch_input_ptr = helperGetTensorPointerWO<int>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);

  Domain full_input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain batch_input_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());

  coord_t sequence_length =
      batch_input_domain.hi()[0] - batch_input_domain.lo()[0] + 1;
  coord_t batch_size =
      batch_input_domain.hi()[1] - batch_input_domain.lo()[1] + 1;

  coord_t full_input_sequence_length =
      batch_input_domain.hi()[0] - batch_input_domain.lo()[0] + 1;
  coord_t full_input_batch_size =
      batch_input_domain.hi()[1] - batch_input_domain.lo()[1] + 1;

  assert(sequence_length == full_input_sequence_length);
  assert(batch_size <= full_input_batch_size);

  // Currently assume continous indices
  assert(bc->num_active_tokens() <= batch_size * sequence_length);
  for (int i = 1; i < bc->num_active_tokens(); i++) {
    auto prev_guid = requestInfo[tokensInfo[i - 1].request_index].request_guid;
    auto guid = requestInfo[tokensInfo[i].request_index].request_guid;
    if (guid == prev_guid) {
      assert(tokensInfo[i].abs_depth_in_request ==
             tokensInfo[i - 1].abs_depth_in_request + 1);
    }
  }
  // keep things simple for now
  assert(batch_input_domain.get_volume() == batch_size * sequence_length);

  // pad inputs if needed (this is really only useful for debugging)
  checkCUDA(cudaMemset(
      batch_input_ptr, 0, batch_input_domain.get_volume() * sizeof(int)));

  auto guid = requestInfo[tokensInfo[0].request_index].request_guid;
  int start_idx = tokensInfo[0].abs_depth_in_request;
  int dst_idx = 0;
  int total_tokens = 0;

  for (size_t i = 1; i <= bc->num_active_tokens(); i++) {
    auto current_guid = requestInfo[tokensInfo[i].request_index].request_guid;
    if (i == bc->num_active_tokens() || current_guid != guid) {

      size_t tokens_to_copy =
          (tokensInfo[i - 1].abs_depth_in_request - start_idx + 1);
      assert(tokens_to_copy > 0);

      int request_index = tokensInfo[i - 1].request_index;
      int token_start_offset =
          bc->requestsInfo[request_index].token_start_offset;
      int num_processing_tokens =
          bc->requestsInfo[request_index].num_tokens_in_batch;
      if (tokens_to_copy > 1 || token_start_offset == 0) {
        // initialization phase
        assert(tokensInfo[i - 1].abs_depth_in_request <
               (token_start_offset + num_processing_tokens));
        int const *input_zc =
            full_input_ptr + (guid * sequence_length) + start_idx;
        int *dst_ptr = batch_input_ptr + dst_idx;
        copy_kernel<<<GET_BLOCKS(tokens_to_copy), CUDA_NUM_THREADS>>>(
            dst_ptr, input_zc, tokens_to_copy);
      } else {
        // incremental phase
        assert(tokensInfo[i - 1].abs_depth_in_request >= token_start_offset);
        assert(tokens_to_copy == 1);

        assert(prev_batch_preds.find(guid) != prev_batch_preds.end());
        int token = prev_batch_preds.at(guid);
        int *dst_ptr = batch_input_ptr + dst_idx;
        cudaMemcpy(dst_ptr,
                   &token,
                   sizeof(FlexFlow::RequestManager::TokenId),
                   cudaMemcpyHostToDevice);
      }
      total_tokens += tokens_to_copy;

      if (i < bc->num_active_tokens()) {
        guid = bc->requestsInfo[bc->tokensInfo[i].request_index].request_guid;
        start_idx = tokensInfo[i].abs_depth_in_request;
      }
      dst_idx = i;
    }
  }
  assert(total_tokens == bc->num_active_tokens());
  /*printf("token_dim: %lli, sequence_length: %lli, batch_size: %lli\n",
  token_dim, sequence_length, batch_size); printf("total_tokens: %lu\n",
  total_tokens); printf("guid: %lu\n", guid);
  print_tensor<int>(batch_input_ptr,
                      batch_input_domain.get_volume(),
                      "[BatchInput]");*/
  checkCUDA(cudaDeviceSynchronize());
}

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
#include "flexflow/utils/cuda_helper.h"

void DataLoader::load_input(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);

  DataLoaderNextBatchInput const input_struct =
      *((DataLoaderNextBatchInput *)task->args);
  // BatchConfig::SampleIdxs const &meta = input_struct.meta;

  BatchConfig *bc = input_struct.bc;
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
    if (tokensInfo[i].guid == tokensInfo[i - 1].guid) {
      assert(tokensInfo[i].abs_depth_in_request ==
             tokensInfo[i - 1].abs_depth_in_request + 1);
    }
  }
  // keep things simple for now
  assert(batch_input_domain.get_volume() == batch_size * sequence_length);

  // pad inputs if needed (this is really only useful for debugging)
  checkCUDA(cudaMemset(
      batch_input_ptr, 0, batch_input_domain.get_volume() * sizeof(int)));

  size_t guid = tokensInfo[0].guid;
  size_t start_idx = tokensInfo[0].abs_depth_in_request;
  size_t dst_idx = 0;
  size_t total_tokens = 0;

  for (size_t i = 1; i <= bc->num_active_tokens(); i++) {
    if (i == bc->num_active_tokens() || tokensInfo[i].guid != guid) {

      size_t tokens_to_copy =
          (tokensInfo[i - 1].abs_depth_in_request - start_idx + 1);
      // size_t size_to_copy = token_dim * tokens_to_copy;
      assert(tokens_to_copy > 0);

      size_t request_index = tokensInfo[i - 1].request_index;
      size_t token_start_offset =
          bc->requestsInfo[request_index].token_start_offset;
      size_t num_processing_tokens =
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

        /* std::cout << "Looking for guid: " << guid << std::endl;
        std::cout << "prev_batch_preds: ";
        for (const auto& elem : prev_batch_preds){
            std::cout << elem.first << ":" << elem.second << ", ";
        }
        std::cout << std::endl; */
        assert(prev_batch_preds.find(guid) != prev_batch_preds.end());
        int token = prev_batch_preds.at(guid);
        int *dst_ptr = batch_input_ptr + dst_idx;
        cudaMemcpy(dst_ptr, &token, sizeof(int), cudaMemcpyHostToDevice);
        // copy_kernel<<<GET_BLOCKS(tokens_to_copy),
        // CUDA_NUM_THREADS>>>(dst_ptr, &token, tokens_to_copy);
        //  cudaMemcpyAsync(batch_input_ptr + dst_idx * token_dim, &token, 1,
        //  cudaMemcpyHostToDevice);
      }
      total_tokens += tokens_to_copy;

      if (i < bc->num_active_tokens()) {
        guid = tokensInfo[i].guid;
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

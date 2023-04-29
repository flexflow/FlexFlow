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

#include "flexflow/utils/cuda_helper.h"
#include "llama.h"

void DataLoader::load_input(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {

  LLAMAConfig llamaconfig;
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //   SampleIdxs *meta = (SampleIdxs *)task->local_args;

  // DataLoaderNextBatchInput const input_struct =
  //     *((DataLoaderNextBatchInput *)task->args);
  // BatchConfig::SampleIdxs const &meta = input_struct.meta;

  DataLoaderNextBatchInput const input_struct =
      *((DataLoaderNextBatchInput *)task->args);
  BatchConfig *bc = input_struct.bc;

  std::map<size_t, long> const &prev_batch_preds =
      input_struct.prev_batch_preds;

  TensorAccessorR<long, 3> full_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<long, 3> batch_input(regions[1],
                                       task->regions[1],
                                       FID_DATA,
                                       ctx,
                                       runtime,
                                       false /*readOutput*/);
  Domain full_input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain batch_input_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());

  coord_t sequence_length =
      batch_input_domain.hi()[0] - batch_input_domain.lo()[0] + 1;
  coord_t batch_size =
      batch_input_domain.hi()[1] - batch_input_domain.lo()[1] + 1;

  // size_t guid = bc->tokensInfo[0].guid;
  size_t guid = bc->requestsInfo[bc->tokensInfo[0].request_index].guid;
  size_t start_idx = bc->tokensInfo[0].abs_depth_in_request;
  size_t dst_idx = 0;

  for (int i = 0; i <= bc->num_active_tokens(); i++) {
    size_t current_guid =
        bc->requestsInfo[bc->tokensInfo[i].request_index].guid;
    if (i == bc->num_active_tokens() || current_guid != guid) {
      size_t tokens_to_copy =
          (bc->tokensInfo[i - 1].abs_depth_in_request - start_idx + 1);

      size_t request_index = bc->tokensInfo[i - 1].request_index;
      size_t token_start_offset =
          bc->requestsInfo[request_index].token_start_offset;

      std::cout << "size to copy:  " << tokens_to_copy
                << ", start offset: " << token_start_offset << "\n";
      if (tokens_to_copy > 1 || token_start_offset == 0) {
        // token pos < init length, the init length is the input sentence length
        // so this is the initial input, load from file.
        size_t copy_start_index = guid * llamaconfig.sentence_len;
        std::cout << "copy index:  " << copy_start_index << "\n";
        copy_kernel<<<GET_BLOCKS(tokens_to_copy), CUDA_NUM_THREADS>>>(
            batch_input.ptr + dst_idx,
            full_input.ptr + copy_start_index,
            tokens_to_copy);
        std::cout << "------------req---------------: " << guid << "\n";
        for (int i = 0; i < 8; i++) {
          std::cout << "value: " << full_input.ptr[copy_start_index + i]
                    << std::endl;
        }
        std::cout << "dst index: " << dst_idx << "\n";

      } else {
        // for token by token generating, get token from the previous inference.

        long token = prev_batch_preds.at(guid);
        // std::cout << "next iter  " << meta.token_indexes[i -
        // 1].token_position
        //           << ", dst_idx: " << dst_idx << ", token:" << token << "\n";

        std::cout << "next iter  " << bc->tokensInfo[i - 1].abs_depth_in_request
                  << ", dst_idx: " << dst_idx << ", token:" << token << "\n";
        long *dst_ptr = batch_input.ptr + dst_idx;

        cudaMemcpy(dst_ptr, &token, sizeof(long), cudaMemcpyHostToDevice);
      }

      if (i < bc->num_active_tokens()) {
        guid = bc->requestsInfo[bc->tokensInfo[i].request_index].guid;
        // guid = bc->tokensInfo[i].guid;
        start_idx = bc->tokensInfo[i].abs_depth_in_request;
      }
      dst_idx = i;
    }
  }

  // copy 1 token from each batch
  //  FIXME: currently assume continous indices
  // size_t guid = meta.guids[0];
  // size_t start_idx = meta.token_indexes[0].token_position;
  // size_t dst_idx = 0;

  std::cout << "load input finished....." << std::endl;
}

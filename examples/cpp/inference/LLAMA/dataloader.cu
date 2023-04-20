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

  DataLoaderNextBatchInput const input_struct =
      *((DataLoaderNextBatchInput *)task->args);
  BatchConfig::SampleIdxs const &meta = input_struct.meta;
  std::map<size_t, Prediction_result> const &prev_batch_preds =
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

  // copy 1 token from each batch
  //  FIXME: currently assume continous indices
  size_t guid = meta.guids[0];
  size_t start_idx = meta.token_indexes[0].token_position;
  size_t dst_idx = 0;

  std::cout << "num samples " << meta.num_samples << "\n";

  for (size_t i = 0; i <= meta.num_samples; i++) {

    // if the first token in one request
    if (i == meta.num_samples || meta.guids[i] != guid) {
      size_t tokens_to_copy =
          (meta.token_indexes[i - 1].token_position - start_idx + 1);
      std::cout << "size to copy:  " << tokens_to_copy << "\n";

      if (tokens_to_copy > 1 || meta.token_indexes[i - 1].token_position <
                                    meta.token_indexes[i - 1].initial_length) {
        // token pos < init length, the init length is the input sentence length
        // so this is the initial input, load from file.

        size_t copy_start_index = guid * llamaconfig.sentence_len;
        std::cout << "copy index:  " << copy_start_index << "\n";
        copy_kernel<<<GET_BLOCKS(tokens_to_copy), CUDA_NUM_THREADS>>>(
            batch_input.ptr + dst_idx,
            full_input.ptr + copy_start_index,
            tokens_to_copy);

        std::cout << "------------req---------------: " << guid << "\n";
        if (guid == 0) {
          std::cout << "guid: " << meta.guids[i] << ", i: " << i << std::endl;
        }
        for (int i = 0; i < 8; i++) {
          std::cout << "value: " << full_input.ptr[copy_start_index + i]
                    << std::endl;
        }
        std::cout << "dst index: " << dst_idx << "\n";

      } else {
        // for token by token generating, get token from the previous inference.
        //generate token based on the sub request size
        // int sub_request_size = bc->
        // long token = prev_batch_preds.at(guid).result;
        // //store the probs into accumulate table

        // //where token is from 
        // std::cout << "next iter  " << meta.token_indexes[i - 1].token_position
        //           << ", dst_idx: " << dst_idx << ", token:" << token << "\n";
        // long *dst_ptr = batch_input.ptr + dst_idx;

        // cudaMemcpy(dst_ptr, &token, sizeof(long), cudaMemcpyHostToDevice);
      }

      // update for next req
      if (i < meta.num_samples) {
        guid = meta.guids[i];
        start_idx = meta.token_indexes[i].token_position;
      }
      dst_idx = i;
    }
  }

  std::cout << "load input finished....." << std::endl;
}

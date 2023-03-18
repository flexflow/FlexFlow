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
  SampleIdxs *meta = (SampleIdxs *)task->local_args;
  if (meta->num_samples == 0) {
    return;
  }
  float const *full_input_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float *batch_input_ptr = helperGetTensorPointerWO<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);

  Domain full_input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain batch_input_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());

  coord_t token_dim =
      batch_input_domain.hi()[0] - batch_input_domain.lo()[0] + 1;
  coord_t sequence_length =
      batch_input_domain.hi()[1] - batch_input_domain.lo()[1] + 1;
  coord_t batch_size =
      batch_input_domain.hi()[2] - batch_input_domain.lo()[2] + 1;
  coord_t full_input_sequence_length =
      full_input_domain.hi()[1] - full_input_domain.lo()[1] + 1;
  assert(sequence_length == full_input_sequence_length);

  // Currently assume continous indices
  assert(meta->num_samples <= batch_size);
  for (int i = 1; i < meta->num_samples; i++) {
    if (meta->guids[i] == meta->guids[i - 1]) {
      assert(meta->idxs[i] == meta->idxs[i - 1] + 1);
    }
  }
  // keep things simple for now
  assert(batch_input_domain.get_volume() ==
         batch_size * sequence_length * token_dim);

  // pad inputs if needed (this is really only useful for debugging)
  checkCUDA(cudaMemset(
      batch_input_ptr, 0, batch_input_domain.get_volume() * sizeof(float)));

  if (!meta->incremental_mode) {
    size_t num_requests = 0;
    size_t guid;
    for (size_t i = 0; i < meta->num_samples; i++) {
      if (i == 0 || meta->guids[i] != guid) {
        guid = meta->guids[0];
        num_requests++;
      }
    }

    coord_t start_idx = meta->guids[0];
    assert(batch_input_domain.get_volume() % batch_size == 0);
    size_t size_to_copy = token_dim * sequence_length * num_requests;
    float const *input_zc =
        full_input_ptr + start_idx * token_dim * sequence_length;
    copy_kernel<<<GET_BLOCKS(size_to_copy), CUDA_NUM_THREADS>>>(
        batch_input_ptr, input_zc, size_to_copy);
    checkCUDA(cudaDeviceSynchronize());
    return;
  }

  size_t guid = meta->guids[0];
  size_t start_idx = meta->idxs[0];
  size_t dst_idx = 0;
  size_t total_tokens = 0;
  for (size_t i = 1; i <= meta->num_samples; i++) {
    if (i == meta->num_samples || meta->guids[i] != guid) {
      size_t size_to_copy = token_dim * (meta->idxs[i - 1] - start_idx + 1);
      total_tokens += size_to_copy / token_dim;
      float const *input_zc = full_input_ptr +
                              (guid * token_dim * sequence_length) +
                              start_idx * token_dim;
      float *dst_ptr = batch_input_ptr + dst_idx * token_dim;
      copy_kernel<<<GET_BLOCKS(size_to_copy), CUDA_NUM_THREADS>>>(
          dst_ptr, input_zc, size_to_copy);
      if (i < meta->num_samples) {
        guid = meta->guids[i];
        start_idx = meta->idxs[i];
      }
      dst_idx = i;
    }
  }
  assert(total_tokens == meta->num_samples);
  checkCUDA(cudaDeviceSynchronize());
}

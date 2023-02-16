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
#include "moe.h"

void DataLoader::load_input(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  SampleIdxs *meta = (SampleIdxs *)task->local_args;
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

  // FIXME: currently assume continous indices
  assert(meta->num_samples <= batch_size);
  for (int i = 1; i < meta->num_samples; i++) {
    assert(meta->idxs[i] == meta->idxs[0] + i);
  }
  // pad inputs if needed (this is really only useful for debugging)
  if (meta->num_samples < batch_size) {
    checkCUDA(cudaMemset(batch_input_ptr +
                             token_dim * sequence_length * meta->num_samples,
                         0,
                         token_dim * sequence_length *
                             (batch_size - meta->num_samples) * sizeof(float)));
  }
  coord_t start_idx = meta->idxs[0];
  assert(batch_input_domain.get_volume() % token_dim * sequence_length *
             batch_size ==
         0);
  assert(batch_input_domain.get_volume() % batch_size == 0);
  size_t size_to_copy =
      (batch_input_domain.get_volume() / batch_size) * meta->num_samples;
  float const *input_zc =
      full_input_ptr + start_idx * token_dim * sequence_length;
  copy_kernel<<<GET_BLOCKS(size_to_copy), CUDA_NUM_THREADS>>>(
      batch_input_ptr, input_zc, size_to_copy);
  checkCUDA(cudaDeviceSynchronize());
}

void DataLoader::load_label(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  SampleIdxs *meta = (SampleIdxs *)task->local_args;
  int const *full_label_ptr = helperGetTensorPointerRO<int>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  int *batch_label_ptr = helperGetTensorPointerWO<int>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  Domain full_label_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain batch_label_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  coord_t label_dim =
      batch_label_domain.hi()[0] - batch_label_domain.lo()[0] + 1;
  coord_t sequence_length =
      batch_label_domain.hi()[1] - batch_label_domain.lo()[1] + 1;
  coord_t batch_size =
      batch_label_domain.hi()[2] - batch_label_domain.lo()[2] + 1;
  // FIXME: currently assume continous indices
  assert(meta->num_samples <= batch_size);
  for (int i = 1; i < meta->num_samples; i++) {
    assert(meta->idxs[i] == meta->idxs[0] + i);
  }
  if (meta->num_samples < batch_size) {
    checkCUDA(cudaMemset(batch_label_ptr +
                             label_dim * sequence_length * meta->num_samples,
                         0,
                         label_dim * sequence_length *
                             (batch_size - meta->num_samples) * sizeof(int)));
  }
  assert(batch_label_domain.get_volume() % label_dim * sequence_length *
             batch_size ==
         0);
  assert(batch_label_domain.get_volume() % batch_size == 0);
  coord_t start_idx = meta->idxs[0];
  size_t size_to_copy =
      (batch_label_domain.get_volume() / batch_size) * meta->num_samples;
  int const *input_zc = acc_full_label.ptr + meta->idxs[0];
  copy_kernel<<<GET_BLOCKS(size_to_copy), CUDA_NUM_THREADS>>>(
      batch_label_ptr, input_zc, size_to_copy);
  checkCUDA(cudaDeviceSynchronize());
}

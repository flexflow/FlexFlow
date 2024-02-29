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

#include "flexflow/dataloader.h"
#include "flexflow/utils/cuda_helper.h"

using namespace Legion;
using namespace FlexFlow;

template <typename DT>
void SingleDataLoader::load_input(Task const *task,
                                  std::vector<PhysicalRegion> const &regions,
                                  Context ctx,
                                  Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  SampleIdxs *meta = (SampleIdxs *)task->local_args;
  Domain full_input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain batch_input_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  const DT *full_input_ptr = helperGetTensorPointerRO<DT>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  DT *batch_input_ptr = helperGetTensorPointerWO<DT>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);

  // add one dim since the batch input has a leading replica dim
  int num_dims = full_input_domain.get_dim();
  assert(num_dims + 1 == batch_input_domain.get_dim());
  // assert the leading replica dim has a degree of one
  //   assert(batch_input_domain.hi()[num_dims] ==
  //          batch_input_domain.lo()[num_dims]);
  coord_t batch_size = batch_input_domain.hi()[num_dims - 1] -
                       batch_input_domain.lo()[num_dims - 1] + 1;

  coord_t replicate_num =
      batch_input_domain.hi()[num_dims] - batch_input_domain.lo()[num_dims] + 1;
  coord_t num_elements_per_batch = batch_input_domain.get_volume() / batch_size;
  // FIXME: currently assume continous indices
  assert(batch_size == meta->num_samples);
  for (int i = 1; i < batch_size; i++) {
    assert(meta->idxs[i] == meta->idxs[0] + i);
  }
  coord_t start_idx = meta->idxs[0];
  const DT *input_zc = full_input_ptr + start_idx * num_elements_per_batch;
  // const int point = task->index_point.point_data[0];
  // printf("Load batch point %d, start_idx %ld, ptr %p\n", point, start_idx,
  // input_zc);
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  // printf("ptr(%p, %p), idx0 %d nb_elements_per_batch %d, batch_size %d,
  // %d\n", acc_full_input.ptr, input_zc, start_idx, num_elements_per_batch,
  // batch_size, start_idx * num_elements_per_batch);
  assert(batch_input_domain.get_volume() % replicate_num == 0);
  copy_kernel_with_replicate<DT>
      <<<GET_BLOCKS(batch_input_domain.get_volume()),
         CUDA_NUM_THREADS,
         0,
         stream>>>(batch_input_ptr,
                   input_zc,
                   batch_input_domain.get_volume() / replicate_num,
                   batch_input_domain.get_volume());
  checkCUDA(cudaDeviceSynchronize());
}

template void SingleDataLoader::load_input<float>(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime);
template void SingleDataLoader::load_input<int32_t>(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime);
template void SingleDataLoader::load_input<int64_t>(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime);

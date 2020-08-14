/* Copyright 2019 Stanford
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

#include "alexnet.h"
#include "cuda_helper.h"

void DataLoader::load_input(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx,
                            Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  SampleIdxs* meta = (SampleIdxs*) task->local_args;
  TensorAccessorR<float, 4> acc_full_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 4> acc_batch_input(
      regions[1], task->regions[1], FID_DATA, ctx, runtime, false/*readOutput*/);
  coord_t batch_size = acc_batch_input.rect.hi[3] - acc_batch_input.rect.lo[3] + 1;
  coord_t channels = acc_batch_input.rect.hi[2] - acc_batch_input.rect.lo[2] + 1;
  coord_t height = acc_batch_input.rect.hi[1] - acc_batch_input.rect.lo[1] + 1;
  coord_t width = acc_batch_input.rect.hi[0] - acc_batch_input.rect.lo[0] + 1;
  //FIXME: currently assume continous indices
  assert(batch_size == meta->num_samples);
  for (int i = 1; i < batch_size; i++)
    assert(meta->idxs[i] == meta->idxs[0] + i);
  coord_t start_idx = meta->idxs[0];
  const float* input_zc = acc_full_input.ptr + start_idx * channels * height * width;
  copy_kernel<<<GET_BLOCKS(acc_batch_input.rect.volume()), CUDA_NUM_THREADS>>>(
      acc_batch_input.ptr, input_zc, acc_batch_input.rect.volume());
  checkCUDA(cudaDeviceSynchronize());
}

void DataLoader::load_label(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx,
                            Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  SampleIdxs* meta = (SampleIdxs*) task->local_args;
  TensorAccessorR<int, 2> acc_full_label(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<int, 2> acc_batch_label(
      regions[1], task->regions[1], FID_DATA, ctx, runtime, false/*readOutput*/);
  int batch_size = acc_batch_label.rect.hi[1] - acc_batch_label.rect.lo[1] + 1;
  //FIXME: currently assume continous indices
  assert(batch_size == meta->num_samples);
  for (int i = 1; i < batch_size; i++)
    assert(meta->idxs[i] == meta->idxs[0] + i);
  const int* input_zc = acc_full_label.ptr + meta->idxs[0];
  copy_kernel<<<GET_BLOCKS(acc_batch_label.rect.volume()), CUDA_NUM_THREADS>>>(
    acc_batch_label.ptr, input_zc, acc_batch_label.rect.volume());
  checkCUDA(cudaDeviceSynchronize());
}


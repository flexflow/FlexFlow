/* Copyright 2021 Stanford, Facebook
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

#include "transformer.h"
#include "cuda_helper.h"

void DataLoader::load_input(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx,
                            Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  SampleIdxs* meta = (SampleIdxs*) task->local_args;
  TensorAccessorR<float, 3> acc_full_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 3> acc_batch_input(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  int batch_size = acc_batch_input.rect.hi[2] - acc_batch_input.rect.lo[2] + 1;
  int embed_size = acc_batch_input.rect.hi[0] - acc_batch_input.rect.lo[0] + 1;
  int seq_length = acc_batch_input.rect.hi[1] - acc_batch_input.rect.lo[1] + 1;
  assert(acc_batch_input.rect.hi[0] == acc_full_input.rect.hi[0]);
  assert(acc_batch_input.rect.lo[0] == acc_full_input.rect.lo[0]);
  assert(acc_batch_input.rect.hi[1] == acc_full_input.rect.hi[1]);
  assert(acc_batch_input.rect.lo[1] == acc_full_input.rect.lo[1]);

  float* input_zc;
  checkCUDA(cudaHostAlloc(&input_zc, sizeof(float) * acc_batch_input.rect.volume(),
                          cudaHostAllocPortable | cudaHostAllocMapped));
  assert(batch_size == meta->num_samples);
  for (int i = 0; i < batch_size; i++) {
    int base_offset = meta->idxs[i] * embed_size * seq_length;
    for (int j = 0; j < embed_size*seq_length; j++)
      input_zc[i*embed_size*seq_length+j] = acc_full_input.ptr[base_offset+j];
  }
  checkCUDA(cudaMemcpy(acc_batch_input.ptr, input_zc,
                       sizeof(float) * acc_batch_input.rect.volume(),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaFreeHost(input_zc));
}


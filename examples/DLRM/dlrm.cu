/* Copyright 2020 Stanford, Facebook
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

#include "dlrm.h"
#include "cuda_helper.h"

void DataLoader::load_sparse_input(const Task *task,
                                   const std::vector<PhysicalRegion> &regions,
                                   Context ctx,
                                   Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  int hash = *((int*) task->args);
  int num_sparse_inputs = hash / 1000;
  int my_input_idx = hash % 1000;
  SampleIdxs* meta = (SampleIdxs*) task->local_args;
  TensorAccessorR<int64_t, 2> acc_full_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<int64_t, 2> acc_batch_input(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  int batch_size = acc_batch_input.rect.hi[1] - acc_batch_input.rect.lo[1] + 1;
  int in_dim = acc_batch_input.rect.hi[0] - acc_batch_input.rect.lo[0] + 1;
  assert(acc_full_input.rect.hi[0]-acc_full_input.rect.lo[0]+1 == num_sparse_inputs * in_dim);
  int64_t* input_zc;
  checkCUDA(cudaHostAlloc(&input_zc, sizeof(int64_t) * acc_batch_input.rect.volume(),
                          cudaHostAllocPortable | cudaHostAllocMapped));
  assert(batch_size == meta->num_samples);
  for (int i = 0; i < batch_size; i++) {
    int full_offset = meta->idxs[i] * num_sparse_inputs * in_dim + my_input_idx * in_dim;
    int batch_offset = i * in_dim;
    assert(full_offset + in_dim <= (int)acc_full_input.rect.volume());
    for (int j = 0; j < in_dim; j++) {
      input_zc[batch_offset+j] = acc_full_input.ptr[full_offset+j];
    }
  }
  checkCUDA(cudaMemcpy(acc_batch_input.ptr, input_zc,
                       sizeof(int64_t) * acc_batch_input.rect.volume(),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaFreeHost(input_zc));
  checkCUDA(cudaDeviceSynchronize());
  //print_tensor<2, int>(acc_batch_input.ptr, acc_batch_input.rect, "[DataLoader:load_sparse]");
}

void DataLoader::load_dense_input(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx,
                                  Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  SampleIdxs* meta = (SampleIdxs*) task->local_args;
  TensorAccessorR<float, 2> acc_full_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 2> acc_batch_input(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  int batch_size = acc_batch_input.rect.hi[1] - acc_batch_input.rect.lo[1] + 1;
  int num_feats = acc_batch_input.rect.hi[0] - acc_batch_input.rect.lo[0] + 1;
  assert(acc_batch_input.rect.hi[0] == acc_full_input.rect.hi[0]);
  assert(acc_batch_input.rect.lo[0] == acc_full_input.rect.lo[0]);
  float* input_zc;
  checkCUDA(cudaHostAlloc(&input_zc, sizeof(float) * acc_batch_input.rect.volume(),
                          cudaHostAllocPortable | cudaHostAllocMapped));
  assert(batch_size == meta->num_samples);
  for (int i = 0; i < batch_size; i++) {
    int base_offset = meta->idxs[i] * num_feats;
    for (int j = 0; j < num_feats; j++)
      input_zc[i*num_feats+j] = acc_full_input.ptr[base_offset+j];
  }
  checkCUDA(cudaMemcpy(acc_batch_input.ptr, input_zc,
                       sizeof(float) * acc_batch_input.rect.volume(),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaFreeHost(input_zc));
}

void DataLoader::load_label(const Task *task,
                            const std::vector<PhysicalRegion>& regions,
                            Context ctx,
                            Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  SampleIdxs* meta = (SampleIdxs*) task->local_args;
  TensorAccessorR<float, 2> acc_full_label(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 2> acc_batch_label(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  int batch_size = acc_batch_label.rect.hi[1] - acc_batch_label.rect.lo[1] + 1;
  int num_label = acc_batch_label.rect.hi[0] - acc_batch_label.rect.lo[0] + 1;
  assert(num_label == 1); // Kaggle dataset a has single label
  assert(acc_batch_label.rect.hi[0] == acc_full_label.rect.hi[0]);
  assert(acc_batch_label.rect.lo[0] == acc_full_label.rect.lo[0]);
  float* label_zc;
  checkCUDA(cudaHostAlloc(&label_zc, sizeof(float) * acc_batch_label.rect.volume(),
                          cudaHostAllocPortable | cudaHostAllocMapped));
  assert(batch_size == meta->num_samples);
  for (int i = 0; i < batch_size; i++) {
    int base_offset = meta->idxs[i] * num_label;
    for (int j = 0; j < num_label; j++)
      label_zc[i*num_label+j] = acc_full_label.ptr[base_offset+j];
    //printf("meta->idxs[%d]=%d label=%.2lf\n", i, meta->idxs[i], label_zc[i]);
  }
  checkCUDA(cudaMemcpy(acc_batch_label.ptr, label_zc,
                       sizeof(float) * acc_batch_label.rect.volume(),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaFreeHost(label_zc));
}


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
  SampleIdxs *meta = (SampleIdxs *)task->local_args;

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

  //copy 1 token from each batch
  // FIXME: currently assume continous indices
  assert(meta->num_samples <= batch_size);
  for (int i = 1; i < meta->num_samples; i++) {
    assert(meta->idxs[i] == meta->idxs[0] + i);
  }
  
//   std::cout << "token idx: " << meta->token_idx <<std::endl;
  
  size_t size_to_copy =
      (batch_input_domain.get_volume());

   checkCUDA(cudaMemset(
      batch_input.ptr, 0, batch_input_domain.get_volume() * sizeof(long)));
  

//   std::cout << size_to_copy <<std::endl;
//   std::cout << "load input....."<<std::endl;
  
  size_t index[size_to_copy];
  size_t *cuda_index;

  //-------get index of input-----
  for(int i = 0; i < batch_size; i++){
     index[i] = meta->batch_idx * (llamaconfig.sentence_len * batch_size) + (llamaconfig.sentence_len * i) + meta->token_idx;
  }

  cudaMalloc((void **)&cuda_index, batch_size * sizeof(size_t));
  cudaMemcpy(cuda_index, index, batch_size * sizeof(size_t), cudaMemcpyHostToDevice);

  
  copy_kernel_discrete<<<GET_BLOCKS(size_to_copy), CUDA_NUM_THREADS>>>(
      batch_input.ptr, full_input.ptr, size_to_copy, cuda_index);
  checkCUDA(cudaDeviceSynchronize());

}
/* Copyright 2021 CMU, Facebook, LANL, MIT, and Stanford (alphabetical)
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

#include "parallel_ops/partition.h"
#include "cuda_helper.h"

using namespace Legion;

template<typename T>
void Repartition::forward_kernel(
    const T* input_ptr,
    T* output_ptr,
    size_t num_elements)
{
  checkCUDA(cudaMemcpyAsync(output_ptr, input_ptr,
      num_elements * sizeof(T),
      cudaMemcpyDeviceToDevice));
}

/*static*/
void Repartition::forward_task(
    const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  Domain input_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  assert(output_domain == input_domain);

  const float* input_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* output_ptr = helperGetTensorPointerWO<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);

#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  //checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  forward_kernel<float>(input_ptr, output_ptr, output_domain.get_volume());
}

template<typename T>
void Repartition::backward_kernel(
    const T* output_grad_ptr,
    T* input_grad_ptr,
    size_t num_elements)
{
  add_kernel<T><<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS>>>(
      input_grad_ptr, output_grad_ptr, num_elements);
}

void Repartition::backward_task(
    const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  Domain output_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain input_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  assert(output_grad_domain == input_grad_domain);

  const float* output_grad_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* input_grad_ptr = helperGetTensorPointerRW<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);

#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  //checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  backward_kernel<float>(output_grad_ptr, input_grad_ptr, output_grad_domain.get_volume());
}

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

#include "parallel_ops/reduction.h"
#include "cuda_helper.h"

using namespace Legion;

template<typename T>
__global__
void reduction_forward_kernel(
    const T* input_ptr,
    T* output_ptr,
    size_t num_elements,
    size_t num_replicas)
{
  CUDA_KERNEL_LOOP(i, num_elements)
  {
    output_ptr[i] = input_ptr[i];
    for (size_t j = 1; j < num_replicas; j++)
      output_ptr[i] += input_ptr[i + j * num_elements];
  }
}

template<typename T>
void Reduction::forward_kernel(
    const T* input_ptr,
    T* output_ptr,
    size_t num_elements,
    size_t num_replicas)
{
  size_t total_elements = num_elements * num_replicas;
  reduction_forward_kernel<T><<<GET_BLOCKS(total_elements), CUDA_NUM_THREADS>>>(
      input_ptr, output_ptr, num_elements, num_replicas);
}

/*static*/
void Reduction::forward_task(
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
  // Currently only support the outter most dimension
  for (int i = 0; i < output_domain.get_dim()-1; i++) {
    assert(output_domain.lo()[i] == input_domain.lo()[i]);
    assert(output_domain.hi()[i] == input_domain.hi()[i]);
  }
  size_t num_elements = output_domain.get_volume();
  size_t num_replicas = input_domain.get_volume() / num_elements;
  const float* input_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* output_ptr = helperGetTensorPointerRW<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);

#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  //checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  forward_kernel<float>(input_ptr, output_ptr, num_elements, num_replicas);
}

template<typename T>
void Reduction::backward_kernel(
    const T* output_grad_ptr,
    T* input_grad_ptr,
    size_t num_elements)
{
  checkCUDA(cudaMemcpyAsync(input_grad_ptr, output_grad_ptr,
      num_elements * sizeof(T),
      cudaMemcpyDeviceToDevice));
}


void Reduction::backward_task(
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
  assert(input_grad_domain.get_volume() == output_grad_domain.get_volume());
  const float* output_grad_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* input_grad_ptr = helperGetTensorPointerWO<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);

#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  //checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  backward_kernel<float>(output_grad_ptr, input_grad_ptr,
      output_grad_domain.get_volume());
}



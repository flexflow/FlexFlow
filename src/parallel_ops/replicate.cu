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

#include "model.h"
#include "cuda_helper.h"

using namespace Legion;

template<typename T>
void Replicate::forward_kernel(
    const T* input_ptr,
    T* output_ptr,
    size_t num_elements)
{
  checkCUDA(cudaMemcpyAsync(output_ptr, input_ptr,
      num_elements * sizeof(T),
      cudaMemcpyDeviceToDevice));
}

void Replicate::forward_task(
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
  assert(input_domain == output_domain);
  const float* input_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* output_ptr = helperGetTensorPointerRW<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  //checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  forward_kernel<float>(input_ptr, output_ptr, input_domain.get_volume());
}

template<typename T>
__global__
void replicate_backward_kernel(
    const T* input_ptr,
    T* output_ptr,
    size_t num_elements,
    size_t num_replicas)
{
  CUDA_KERNEL_LOOP(i, num_elements)
  {
    for (size_t j = 0; j < num_replicas; j++)
      output_ptr[i] += input_ptr[i + j * num_elements];
  }
}

template<typename T>
void Replicate::backward_kernel(
    const T* output_grad_ptr,
    T* input_grad_ptr,
    size_t num_elements,
    size_t num_replicas)
{
  size_t total_elements = num_elements * num_replicas;
  replicate_backward_kernel<T><<<GET_BLOCKS(total_elements), CUDA_NUM_THREADS>>>(
      output_grad_ptr, input_grad_ptr, num_elements, num_replicas);
}

void Replicate::backward_task(
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
  // Currently only support the outter most dimension
  for (int i = 0; i < output_grad_domain.get_dim()-1; i++) {
    assert(output_grad_domain.lo()[i] == input_grad_domain.lo()[i]);
    assert(output_grad_domain.hi()[i] == input_grad_domain.hi()[i]);
  }
  size_t num_elements = input_grad_domain.get_volume();
  size_t num_replicas = output_grad_domain.get_volume() / num_elements;
  const float* output_grad_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* input_grad_ptr = helperGetTensorPointerRW<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);

#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  //checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  backward_kernel<float>(output_grad_ptr, input_grad_ptr,
      num_elements, num_replicas);
}

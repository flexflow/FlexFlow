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

#include "flexflow/parallel_ops/kernels/replicate_kernels.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

ReplicateMeta::ReplicateMeta(FFHandler handle, Replicate const *repl)
    : OpMeta(handle, repl) {}

namespace Kernels {
namespace Replicate {

template <typename T>
void forward_kernel(T const *input_ptr, T *output_ptr, size_t num_elements) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDA(cudaMemcpyAsync(output_ptr,
                            input_ptr,
                            num_elements * sizeof(T),
                            cudaMemcpyDeviceToDevice,
                            stream));
}

template <typename T>
__global__ void replicate_backward_kernel(T const *input_ptr,
                                          T *output_ptr,
                                          size_t num_elements,
                                          size_t num_replicas) {
  CUDA_KERNEL_LOOP(i, num_elements) {
    for (size_t j = 0; j < num_replicas; j++) {
      output_ptr[i] += input_ptr[i + j * num_elements];
    }
  }
}

template <typename T>
void backward_kernel(T const *output_grad_ptr,
                     T *input_grad_ptr,
                     size_t num_elements,
                     size_t num_replicas) {
  size_t total_elements = num_elements * num_replicas;
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  replicate_backward_kernel<T>
      <<<GET_BLOCKS(total_elements), CUDA_NUM_THREADS, 0, stream>>>(
          output_grad_ptr, input_grad_ptr, num_elements, num_replicas);
}

template void forward_kernel<float>(float const *input_ptr,
                                    float *output_ptr,
                                    size_t num_elements);
template void forward_kernel<half>(half const *input_ptr,
                                   half *output_ptr,
                                   size_t num_elements);
template __global__ void
    replicate_backward_kernel<float>(float const *input_ptr,
                                     float *output_ptr,
                                     size_t num_elements,
                                     size_t num_replicas);
template void backward_kernel<float>(float const *output_grad_ptr,
                                     float *input_grad_ptr,
                                     size_t num_elements,
                                     size_t num_replicas);

} // namespace Replicate
} // namespace Kernels
} // namespace FlexFlow

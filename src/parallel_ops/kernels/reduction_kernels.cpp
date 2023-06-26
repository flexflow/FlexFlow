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

#include "flexflow/parallel_ops/kernels/reduction_kernels.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

ReductionMeta::ReductionMeta(FFHandler handle, Reduction const *reduct)
    : OpMeta(handle) {}

namespace Kernels {
namespace Reduction {

template <typename T>
__global__ void reduction_forward_kernel(T const *input_ptr,
                                         T *output_ptr,
                                         size_t num_elements,
                                         size_t num_replicas) {
  CUDA_KERNEL_LOOP(i, num_elements) {
    output_ptr[i] = input_ptr[i];
    for (size_t j = 1; j < num_replicas; j++) {
      output_ptr[i] += input_ptr[i + j * num_elements];
    }
  }
}

template <typename T>
void forward_kernel(T const *input_ptr,
                    T *output_ptr,
                    size_t num_elements,
                    size_t num_replicas) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  size_t total_elements = num_elements * num_replicas;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(reduction_forward_kernel<T>),
                     GET_BLOCKS(total_elements),
                     CUDA_NUM_THREADS,
                     0,
                     stream,
                     input_ptr,
                     output_ptr,
                     num_elements,
                     num_replicas);
}

template <typename T>
void backward_kernel(T const *output_grad_ptr,
                     T *input_grad_ptr,
                     size_t num_elements) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDA(hipMemcpyAsync(input_grad_ptr,
                           output_grad_ptr,
                           num_elements * sizeof(T),
                           hipMemcpyDeviceToDevice,
                           stream));
}

template __global__ void reduction_forward_kernel<float>(float const *input_ptr,
                                                         float *output_ptr,
                                                         size_t num_elements,
                                                         size_t num_replicas);
template __global__ void reduction_forward_kernel<half>(half const *input_ptr,
                                                        half *output_ptr,
                                                        size_t num_elements,
                                                        size_t num_replicas);
template void forward_kernel<float>(float const *input_ptr,
                                    float *output_ptr,
                                    size_t num_elements,
                                    size_t num_replicas);
template void forward_kernel<half>(half const *input_ptr,
                                   half *output_ptr,
                                   size_t num_elements,
                                   size_t num_replicas);
template void backward_kernel<float>(float const *output_grad_ptr,
                                     float *input_grad_ptr,
                                     size_t num_elements);
} // namespace Reduction
} // namespace Kernels
} // namespace FlexFlow

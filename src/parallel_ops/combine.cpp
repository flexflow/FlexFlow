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

#include <hip/hip_runtime.h>
#include "flexflow/parallel_ops/combine.h"
#include "flexflow/utils/hip_helper.h"

namespace FlexFlow {

template<typename T>
void Combine::forward_kernel(
    const T* input_ptr,
    T* output_ptr,
    size_t num_elements)
{
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDA(hipMemcpyAsync(output_ptr, input_ptr,
      num_elements * sizeof(T),
      hipMemcpyDeviceToDevice,
      stream));
}

template<typename T>
void Combine::backward_kernel(
    const T* output_grad_ptr,
    T* input_grad_ptr,
    size_t num_elements)
{
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  hipLaunchKernelGGL(HIP_KERNEL_NAME(add_kernel<T>), GET_BLOCKS(num_elements), CUDA_NUM_THREADS, 0, stream, 
      input_grad_ptr, output_grad_ptr, num_elements);
}

CombineMeta::CombineMeta(FFHandler handler)
: OpMeta(handler) {}

template void Combine::forward_kernel<float>(const float* input_ptr, float* output_ptr, size_t num_elements);
template void Combine::forward_kernel<double>(const double* input_ptr, double* output_ptr, size_t num_elements);
template void Combine::forward_kernel<int32_t>(const int32_t* input_ptr, int32_t* output_ptr, size_t num_elements);
template void Combine::forward_kernel<int64_t>(const int64_t* input_ptr, int64_t* output_ptr, size_t num_elements);
template void Combine::backward_kernel<float>(const float* output_grad_ptr, float* input_grad_ptr, size_t num_elements);
template void Combine::backward_kernel<double>(const double* output_grad_ptr, double* input_grad_ptr, size_t num_elements);
template void Combine::backward_kernel<int32_t>(const int32_t* output_grad_ptr, int32_t* input_grad_ptr, size_t num_elements);
template void Combine::backward_kernel<int64_t>(const int64_t* output_grad_ptr, int64_t* input_grad_ptr, size_t num_elements);

}; // namespace FlexFlow

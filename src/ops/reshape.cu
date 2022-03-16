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

#include "flexflow/ops/reshape.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

ReshapeMeta::ReshapeMeta(FFHandler handler)
: OpMeta(handler) {}

/*static*/
template<typename T>
void Reshape::forward_kernel(const T* input_ptr,
                             T* output_ptr,
                             size_t num_elements,
                             cudaStream_t stream)
{
  checkCUDA(cudaMemcpyAsync(output_ptr, input_ptr,
      num_elements * sizeof(T), cudaMemcpyDeviceToDevice, stream));
}

/*static*/
template<typename T>
void Reshape::forward_kernel_wrapper(const T* input_ptr,
                                     T* output_ptr,
                                     size_t num_elements)
{
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  Reshape::forward_kernel<T>(input_ptr, output_ptr, num_elements, stream);
}

/*static*/
template<typename T>
void Reshape::backward_kernel(T* input_grad_ptr,
                              const T* output_grad_ptr,
                              size_t num_elements,
                              cudaStream_t stream)
{
  float alpha = 1.0f;
  apply_add_with_scale<T><<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS, 0, stream>>>(
      input_grad_ptr, output_grad_ptr, num_elements, (T)alpha);
}

/*static*/
template<typename T>
void Reshape::backward_kernel_wrapper(T* input_grad_ptr,
                                      const T* output_grad_ptr,
                                      size_t num_elements)
{
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  Reshape::backward_kernel<T>(input_grad_ptr, output_grad_ptr, num_elements, stream);
}

template void Reshape::forward_kernel<float>(const float* input_ptr, float* output_ptr, size_t num_elements, cudaStream_t stream);
template void Reshape::forward_kernel<double>(const double* input_ptr, double* output_ptr, size_t num_elements, cudaStream_t stream);
template void Reshape::forward_kernel<int32_t>(const int32_t* input_ptr, int32_t* output_ptr, size_t num_elements, cudaStream_t stream);
template void Reshape::forward_kernel<int64_t>(const int64_t* input_ptr, int64_t* output_ptr, size_t num_elements, cudaStream_t stream);

template void Reshape::forward_kernel_wrapper<float>(const float* input_ptr, float* output_ptr, size_t volume);
template void Reshape::forward_kernel_wrapper<double>(const double* input_ptr, double* output_ptr, size_t volume);
template void Reshape::forward_kernel_wrapper<int32_t>(const int32_t* input_ptr, int32_t* output_ptr, size_t volume);
template void Reshape::forward_kernel_wrapper<int64_t>(const int64_t* input_ptr, int64_t* output_ptr, size_t volume);

template void Reshape::backward_kernel<float>(float* input_grad_ptr, const float* output_grad_ptr, size_t num_elements, cudaStream_t stream);
template void Reshape::backward_kernel<double>(double* input_grad_ptr, const double* output_grad_ptr, size_t num_elements, cudaStream_t stream);
template void Reshape::backward_kernel<int32_t>(int32_t* input_grad_ptr, const int32_t* output_grad_ptr, size_t num_elements, cudaStream_t stream);
template void Reshape::backward_kernel<int64_t>(int64_t* input_grad_ptr, const int64_t* output_grad_ptr, size_t num_elements, cudaStream_t stream);

template void Reshape::backward_kernel_wrapper<float>(float* in_grad_ptr, const float* out_grad_ptr, size_t volume);
template void Reshape::backward_kernel_wrapper<double>(double* in_grad_ptr, const double* out_grad_ptr, size_t volume);
template void Reshape::backward_kernel_wrapper<int32_t>(int32_t* in_grad_ptr, const int32_t* out_grad_ptr, size_t volume);
template void Reshape::backward_kernel_wrapper<int64_t>(int64_t* in_grad_ptr, const int64_t* out_grad_ptr, size_t volume);

}; // namespace FlexFlow

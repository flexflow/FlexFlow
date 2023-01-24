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

#include "flexflow/parallel_ops/kernels/partition_kernels.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

RepartitionMeta::RepartitionMeta(FFHandler handler) : OpMeta(handler) {}

namespace Kernels {
namespace Repartition {

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
void backward_kernel(T const *output_grad_ptr,
                     T *input_grad_ptr,
                     size_t num_elements) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  add_kernel<T><<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS, 0, stream>>>(
      input_grad_ptr, output_grad_ptr, num_elements);
}

template void forward_kernel<float>(float const *input_ptr,
                                    float *output_ptr,
                                    size_t num_elements);
template void forward_kernel<double>(double const *input_ptr,
                                     double *output_ptr,
                                     size_t num_elements);
template void forward_kernel<int32_t>(int32_t const *input_ptr,
                                      int32_t *output_ptr,
                                      size_t num_elements);
template void forward_kernel<int64_t>(int64_t const *input_ptr,
                                      int64_t *output_ptr,
                                      size_t num_elements);
template void backward_kernel<float>(float const *output_grad_ptr,
                                     float *input_grad_ptr,
                                     size_t num_elements);
template void backward_kernel<double>(double const *output_grad_ptr,
                                      double *input_grad_ptr,
                                      size_t num_elements);
template void backward_kernel<int32_t>(int32_t const *output_grad_ptr,
                                       int32_t *input_grad_ptr,
                                       size_t num_elements);
template void backward_kernel<int64_t>(int64_t const *output_grad_ptr,
                                       int64_t *input_grad_ptr,
                                       size_t num_elements);

} // namespace Repartition
} // namespace Kernels
} // namespace FlexFlow

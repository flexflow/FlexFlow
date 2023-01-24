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

#include "flexflow/ops/kernels/reshape_kernels.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

ReshapeMeta::ReshapeMeta(FFHandler handler) : OpMeta(handler) {}

namespace Kernels {
namespace Reshape {

template <typename T>
void forward_kernel_wrapper(T const *input_ptr,
                            T *output_ptr,
                            size_t num_elements) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  Internal::forward_kernel<T>(input_ptr, output_ptr, num_elements, stream);
}

template <typename T>
void backward_kernel_wrapper(T *input_grad_ptr,
                             T const *output_grad_ptr,
                             size_t num_elements) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  Internal::backward_kernel<T>(
      input_grad_ptr, output_grad_ptr, num_elements, stream);
}

template void forward_kernel_wrapper<float>(float const *input_ptr,
                                            float *output_ptr,
                                            size_t volume);
template void forward_kernel_wrapper<double>(double const *input_ptr,
                                             double *output_ptr,
                                             size_t volume);
template void forward_kernel_wrapper<int32_t>(int32_t const *input_ptr,
                                              int32_t *output_ptr,
                                              size_t volume);
template void forward_kernel_wrapper<int64_t>(int64_t const *input_ptr,
                                              int64_t *output_ptr,
                                              size_t volume);

template void backward_kernel_wrapper<float>(float *in_grad_ptr,
                                             float const *out_grad_ptr,
                                             size_t volume);
template void backward_kernel_wrapper<double>(double *in_grad_ptr,
                                              double const *out_grad_ptr,
                                              size_t volume);
template void backward_kernel_wrapper<int32_t>(int32_t *in_grad_ptr,
                                               int32_t const *out_grad_ptr,
                                               size_t volume);
template void backward_kernel_wrapper<int64_t>(int64_t *in_grad_ptr,
                                               int64_t const *out_grad_ptr,
                                               size_t volume);

namespace Internal {

template <typename T>
void forward_kernel(T const *input_ptr,
                    T *output_ptr,
                    size_t num_elements,
                    hipStream_t stream) {
  checkCUDA(hipMemcpyAsync(output_ptr,
                           input_ptr,
                           num_elements * sizeof(T),
                           hipMemcpyDeviceToDevice,
                           stream));
}

template <typename T>
void backward_kernel(T *input_grad_ptr,
                     T const *output_grad_ptr,
                     size_t num_elements,
                     hipStream_t stream) {
  float alpha = 1.0f;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(apply_add_with_scale<T>),
                     GET_BLOCKS(num_elements),
                     CUDA_NUM_THREADS,
                     0,
                     stream,
                     input_grad_ptr,
                     output_grad_ptr,
                     num_elements,
                     (T)alpha);
}

template void forward_kernel<float>(float const *input_ptr,
                                    float *output_ptr,
                                    size_t num_elements,
                                    hipStream_t stream);
template void forward_kernel<double>(double const *input_ptr,
                                     double *output_ptr,
                                     size_t num_elements,
                                     hipStream_t stream);
template void forward_kernel<int32_t>(int32_t const *input_ptr,
                                      int32_t *output_ptr,
                                      size_t num_elements,
                                      hipStream_t stream);
template void forward_kernel<int64_t>(int64_t const *input_ptr,
                                      int64_t *output_ptr,
                                      size_t num_elements,
                                      hipStream_t stream);

template void backward_kernel<float>(float *input_grad_ptr,
                                     float const *output_grad_ptr,
                                     size_t num_elements,
                                     hipStream_t stream);
template void backward_kernel<double>(double *input_grad_ptr,
                                      double const *output_grad_ptr,
                                      size_t num_elements,
                                      hipStream_t stream);
template void backward_kernel<int32_t>(int32_t *input_grad_ptr,
                                       int32_t const *output_grad_ptr,
                                       size_t num_elements,
                                       hipStream_t stream);
template void backward_kernel<int64_t>(int64_t *input_grad_ptr,
                                       int64_t const *output_grad_ptr,
                                       size_t num_elements,
                                       hipStream_t stream);

} // namespace Internal
} // namespace Reshape
} // namespace Kernels
} // namespace FlexFlow

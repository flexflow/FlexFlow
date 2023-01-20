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

#include "flexflow/ops/kernels/flat_kernels.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

namespace Kernels {
namespace Flat {

void forward_kernel_wrapper(float const *input_ptr,
                            float *output_ptr,
                            size_t num_elements) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  Internal::forward_kernel(input_ptr, output_ptr, num_elements, stream);
  // checkCUDA(cudaDeviceSynchronize());
}

void backward_kernel_wrapper(float *input_grad_ptr,
                             float const *output_grad_ptr,
                             size_t num_elements) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  Internal::backward_kernel(
      input_grad_ptr, output_grad_ptr, num_elements, stream);
  // checkCUDA(cudaMemcpyAsync(acc_input_grad.ptr, acc_output_grad.ptr,
  //                           acc_input_grad.rect.volume() * sizeof(float),
  //                           cudaMemcpyDeviceToDevice));
  // checkCUDA(cudaDeviceSynchronize());
}

namespace Internal {

void forward_kernel(float const *input_ptr,
                    float *output_ptr,
                    size_t num_elements,
                    cudaStream_t stream) {
  checkCUDA(cudaMemcpyAsync(output_ptr,
                            input_ptr,
                            num_elements * sizeof(float),
                            cudaMemcpyDeviceToDevice,
                            stream));
}

void backward_kernel(float *input_grad_ptr,
                     float const *output_grad_ptr,
                     size_t num_elements,
                     cudaStream_t stream) {
  float alpha = 1.0f;
  apply_add_with_scale<float>
      <<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS, 0, stream>>>(
          input_grad_ptr, output_grad_ptr, num_elements, alpha);
}

} // namespace Internal
} // namespace Flat
} // namespace Kernels
} // namespace FlexFlow

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

#include "internal/device.h"
#include "kernels/accessor.h"
#include "kernels/flat_kernels_gpu.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

void flat_gpu_forward_kernel(cudaStream_t stream,
                             void const *input_ptr,
                             void *output_ptr,
                             size_t num_elements,
                             size_t element_size) {
  checkCUDA(cudaMemcpyAsync(output_ptr,
                            input_ptr,
                            num_elements * element_size,
                            cudaMemcpyDeviceToDevice,
                            stream));
}

void flat_gpu_backward_kernel(cudaStream_t stream,
                              float const *output_grad_ptr,
                              float *input_grad_ptr,
                              size_t num_elements) {

  float alpha = 1.0f;
  apply_add_with_scale<float>
      <<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS, 0, stream>>>(
          input_grad_ptr, output_grad_ptr, num_elements, alpha);
}

} // namespace FlexFlow

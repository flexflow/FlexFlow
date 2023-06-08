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

#include "kernels/softmax_kernels.h"
#include "kernels/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {
// declare Legion names
using Legion::Domain;

SoftmaxPerDeviceState::SoftmaxPerDeviceState(FFHandler handler,
                                             Softmax const *softmax,
                                             Domain const &input_domain)
    : PerDeviceOpState(handler) {
  checkCUDNN(miopenCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(inputTensor, input_domain));
  dim = softmax->dim;
  profiling = softmax->profiling;
  std::strcpy(op_name, softmax->name);
}

namespace Kernels {
namespace Softmax {

void forward_kernel(hipStream_t stream, SoftmaxPerDeviceState const *m,
                    float const *input_ptr, float *output_ptr) {
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(miopenSoftmaxForward_V2(
      m->handle.dnn, &alpha, m->inputTensor, input_ptr, &beta, m->inputTensor,
      output_ptr, MIOPEN_SOFTMAX_ACCURATE, MIOPEN_SOFTMAX_MODE_CHANNEL));
}

void backward_kernel(hipStream_t stream, float *input_grad_ptr,
                     float const *output_grad_ptr, size_t num_elements) {
  checkCUDA(hipMemcpyAsync(input_grad_ptr, output_grad_ptr,
                           num_elements * sizeof(float),
                           hipMemcpyDeviceToDevice, stream));
}

} // namespace Softmax
} // namespace Kernels
} // namespace FlexFlow

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

#include "flexflow/ops/kernels/softmax_kernels.h"
#include "flexflow/utils/hash_utils.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {
// declare Legion names
using Legion::Domain;

SoftmaxMeta::SoftmaxMeta(FFHandler handler,
                         Softmax const *softmax,
                         Domain const &input_domain)
    : OpMeta(handler) {
  checkCUDNN(miopenCreateTensorDescriptor(&inputTensor));
  // checkCUDNN(cudnnSetTensorDescriptorFromDomain(inputTensor, input_domain));
  checkCUDNN(
      cudnnSetTensorDescriptorFromDomain4SoftMax(inputTensor, input_domain));
  dim = softmax->dim;
  last_layer = softmax->last_layer;
  profiling = softmax->profiling;
  std::strcpy(op_name, softmax->name);
}

namespace Kernels {
namespace Softmax {

void forward_kernel_wrapper(SoftmaxMeta const *m,
                            float const *input_ptr,
                            float *output_ptr) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    hipEventCreate(&t_start);
    hipEventCreate(&t_end);
    hipEventRecord(t_start, stream);
  }
  Internal::forward_kernel(m, input_ptr, output_ptr, stream);
  if (m->profiling) {
    hipEventRecord(t_end, stream);
    checkCUDA(hipEventSynchronize(t_end));
    // print_tensor<float>(acc_input.ptr, acc_input.rect.volume(),
    // "[Softmax:forward:input]"); print_tensor<float>(acc_output.ptr,
    // acc_output.rect.volume(), "[Softmax:forward:output]");
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    hipEventDestroy(t_start);
    hipEventDestroy(t_end);
    log_measure.debug(
        "%s [Softmax] forward time = %.2fms\n", m->op_name, elapsed);
  }
}

void backward_kernel_wrapper(SoftmaxMeta const *m,
                             float *input_grad_ptr,
                             float const *output_grad_ptr,
                             float const *output_ptr,
                             size_t num_elements) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    hipEventCreate(&t_start);
    hipEventCreate(&t_end);
    hipEventRecord(t_start, stream);
  }
  Internal::backward_kernel(
      m, input_grad_ptr, output_grad_ptr, output_ptr, num_elements, stream);
  if (m->profiling) {
    hipEventRecord(t_end, stream);
    checkCUDA(hipEventSynchronize(t_end));
    // print_tensor<float>(acc_output_grad.ptr, acc_output_grad.rect.volume(),
    // "[Softmax:backward:output_grad]");
    // print_tensor<float>(acc_input_grad.ptr, acc_input_grad.rect.volume(),
    // "[Softmax:backward:input_grad]");
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    hipEventDestroy(t_start);
    hipEventDestroy(t_end);
    log_measure.debug("Softmax backward time = %.2fms\n", elapsed);
  }
}

namespace Internal {

void forward_kernel(SoftmaxMeta const *m,
                    float const *input_ptr,
                    float *output_ptr,
                    hipStream_t stream) {
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(miopenSoftmaxForward_V2(m->handle.dnn,
                                     &alpha,
                                     m->inputTensor,
                                     input_ptr,
                                     &beta,
                                     m->inputTensor,
                                     output_ptr,
                                     MIOPEN_SOFTMAX_ACCURATE,
                                     MIOPEN_SOFTMAX_MODE_CHANNEL));
}

void backward_kernel(SoftmaxMeta const *m,
                     float *input_grad_ptr,
                     float const *output_grad_ptr,
                     float const *output_ptr,
                     size_t num_elements,
                     hipStream_t stream) {
  if (m->last_layer) {
    checkCUDA(hipMemcpyAsync(input_grad_ptr,
                             output_grad_ptr,
                             num_elements * sizeof(float),
                             hipMemcpyDeviceToDevice,
                             stream));
  } else {
    float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(miopenSoftmaxBackward_V2(m->handle.dnn,
                                        &alpha,
                                        m->inputTensor,
                                        output_ptr,
                                        m->inputTensor,
                                        output_grad_ptr,
                                        &beta,
                                        m->inputTensor,
                                        input_grad_ptr,
                                        MIOPEN_SOFTMAX_ACCURATE,
                                        MIOPEN_SOFTMAX_MODE_CHANNEL));
  }
}

} // namespace Internal
} // namespace Softmax
} // namespace Kernels
} // namespace FlexFlow

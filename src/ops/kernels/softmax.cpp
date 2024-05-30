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
  checkCUDNN(
      cudnnSetTensorDescriptorFromDomain4SoftMax(inputTensor, input_domain));
  checkCUDNN(miopenCreateTensorDescriptor(&outputTensor));
  checkCUDNN(
      cudnnSetTensorDescriptorFromDomain4SoftMax(outputTensor, input_domain));
  dim = softmax->dim;
  profiling = softmax->profiling;
  inference_debugging = softmax->inference_debugging;
  std::strcpy(op_name, softmax->name);
}

namespace Kernels {
namespace Softmax {

template <typename DT>
void forward_kernel_wrapper(SoftmaxMeta const *m,
                            DT const *input_ptr,
                            DT *output_ptr) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
  }
  Internal::forward_kernel(m, input_ptr, output_ptr, stream);
  if (m->profiling) {
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    // print_tensor<float>(acc_input.ptr, acc_input.rect.volume(),
    // "[Softmax:forward:input]"); print_tensor<float>(acc_output.ptr,
    // acc_output.rect.volume(), "[Softmax:forward:output]");
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
    log_measure.debug(
        "%s [Softmax] forward time = %.2fms\n", m->op_name, elapsed);
  }
}

template <typename DT>
void backward_kernel_wrapper(SoftmaxMeta const *m,
                             DT *input_grad_ptr,
                             DT const *output_grad_ptr,
                             size_t num_elements) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
  }
  Internal::backward_kernel(
      input_grad_ptr, output_grad_ptr, num_elements, stream);
  if (m->profiling) {
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    // print_tensor<float>(acc_output_grad.ptr, acc_output_grad.rect.volume(),
    // "[Softmax:backward:output_grad]");
    // print_tensor<float>(acc_input_grad.ptr, acc_input_grad.rect.volume(),
    // "[Softmax:backward:input_grad]");
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
    log_measure.debug("Softmax backward time = %.2fms\n", elapsed);
  }
}

template void forward_kernel_wrapper<float>(SoftmaxMeta const *m,
                                            float const *input_ptr,
                                            float *output_ptr);
template void forward_kernel_wrapper<half>(SoftmaxMeta const *m,
                                           half const *input_ptr,
                                           half *output_ptr);

template void backward_kernel_wrapper<float>(SoftmaxMeta const *m,
                                             float *input_grad_ptr,
                                             float const *output_grad_ptr,
                                             size_t num_elements);
template void backward_kernel_wrapper<half>(SoftmaxMeta const *m,
                                            half *input_grad_ptr,
                                            half const *output_grad_ptr,
                                            size_t num_elements);

namespace Internal {
template <typename DT>
void forward_kernel(SoftmaxMeta const *m,
                    DT const *input_ptr,
                    DT *output_ptr,
                    hipStream_t stream) {
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(miopenSoftmaxForward_V2(m->handle.dnn,
                                     &alpha,
                                     m->inputTensor,
                                     input_ptr,
                                     &beta,
                                     m->outputTensor,
                                     output_ptr,
                                     MIOPEN_SOFTMAX_ACCURATE,
                                     MIOPEN_SOFTMAX_MODE_CHANNEL));
}

template <typename DT>
void backward_kernel(DT *input_grad_ptr,
                     DT const *output_grad_ptr,
                     size_t num_elements,
                     hipStream_t stream) {
  checkCUDA(hipMemcpyAsync(input_grad_ptr,
                           output_grad_ptr,
                           num_elements * sizeof(DT),
                           hipMemcpyDeviceToDevice,
                           stream));
}

} // namespace Internal
} // namespace Softmax
} // namespace Kernels
} // namespace FlexFlow

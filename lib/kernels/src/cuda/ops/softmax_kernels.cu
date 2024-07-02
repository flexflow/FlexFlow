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

#include "device.h"
#include "kernels/softmax_kernels.h"

namespace FlexFlow {

namespace Kernels {
namespace Softmax {

SoftmaxPerDeviceState init_kernel(PerDeviceFFHandle const &handle,
                                  int dim,
                                  int input_n,
                                  int input_c,
                                  int input_h,
                                  int input_w) {
  ffTensorDescriptor_t inputTensor;

  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        input_n,
                                        input_c,
                                        input_h,
                                        input_w));

  SoftmaxPerDeviceState per_device_state = {handle, inputTensor, dim};
  return per_device_state;
}

void forward_kernel(cudaStream_t stream,
                    SoftmaxPerDeviceState const &m,
                    float const *input_ptr,
                    float *output_ptr) {
  checkCUDNN(cudnnSetStream(m.handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnSoftmaxForward(m.handle.dnn,
                                 CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_CHANNEL,
                                 &alpha,
                                 m.inputTensor,
                                 input_ptr,
                                 &beta,
                                 m.inputTensor,
                                 output_ptr));
}

void backward_kernel(cudaStream_t stream,
                     float *input_grad_ptr,
                     float const *output_grad_ptr,
                     size_t num_elements) {

  checkCUDA(cudaMemcpyAsync(input_grad_ptr,
                            output_grad_ptr,
                            num_elements * sizeof(float),
                            cudaMemcpyDeviceToDevice,
                            stream));
}

} // namespace Softmax
} // namespace Kernels
} // namespace FlexFlow

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

#include "kernels/pool_2d_kernels.h"
#include "device.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

namespace Kernels {
namespace Pool2D {

Pool2DPerDeviceState init_kernel(PerDeviceFFHandle handle,
                                 optional<Activation> activation,
                                 int input_w,
                                 int input_h,
                                 int input_c,
                                 int input_n,
                                 int output_w,
                                 int output_h,
                                 int output_c,
                                 int output_n,
                                 int pad_h,
                                 int pad_w,
                                 int kernel_h,
                                 int kernel_w,
                                 int stride_h,
                                 int stride_w,
                                 PoolOp pool_type) {
  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t outputTensor;
  ffPoolingDescriptor_t poolDesc;
  ffActivationDescriptor_t actiDesc;

  checkCUDNN(miopenCreateTensorDescriptor(&inputTensor));
  checkCUDNN(miopenCreateTensorDescriptor(&outputTensor));
  checkCUDNN(miopenCreatePoolingDescriptor(&poolDesc));
  checkCUDNN(miopenCreateActivationDescriptor(&actiDesc));

  checkCUDNN(miopenSet4dTensorDescriptor(
      inputTensor, miopenFloat, input_n, input_c, input_h, input_w));
  cudnnPoolingMode_t mode;
  if (pool_type == PoolOp::MAX) {
    mode = MIOPEN_POOLING_MAX;
  } else {
    assert(pool_type == PoolOp::AVG);
    mode = MIOPEN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  }

  checkCUDNN(miopenSetPooling2dDescriptor(
      poolDesc, mode, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));

  int n, c, h, w;
  checkCUDNN(miopenGetPooling2dForwardOutputDim(
      poolDesc, inputTensor, &n, &c, &h, &w));
  assert(n == output_n);
  assert(c == output_c);
  assert(h == output_h);
  assert(w == output_w);

  checkCUDNN(
      miopenSet4dTensorDescriptor(outputTensor, miopenFloat, n, c, h, w));
  bool relu = false;
  if (activation == Activation::RELU) {
    relu = true;
  }
  Pool2DPerDeviceState state = {
      handle,
      inputTensor,
      outputTensor,
      actiDesc,
      poolDesc,
      relu,
  };
  return state;
}

void forward_kernel(hipStream_t stream,
                    Pool2DPerDeviceState const &m,
                    void const *input_ptr,
                    void *output_ptr) {
  checkCUDNN(miopenSetStream(m.handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(miopenPoolingForward(m.handle.dnn,
                                  m.poolDesc,
                                  &alpha,
                                  m.inputTensor,
                                  input_ptr,
                                  &beta,
                                  m.outputTensor,
                                  output_ptr,
                                  true,
                                  m.handle.workSpace,
                                  m.handle.workSpaceSize));
}

void backward_kernel(hipStream_t stream,
                     Pool2DPerDeviceState const &m,
                     void const *input_ptr,
                     void *input_grad_ptr,
                     void const *output_ptr,
                     void const *output_grad_ptr) {

  checkCUDNN(miopenSetStream(m.handle.dnn, stream));

  float alpha = 1.0f;
  checkCUDNN(miopenPoolingBackward(m.handle.dnn,
                                   m.poolDesc,
                                   &alpha,
                                   m.outputTensor,
                                   output_ptr,
                                   m.outputTensor,
                                   output_grad_ptr,
                                   m.inputTensor,
                                   input_ptr,
                                   &beta,
                                   m.inputTensor,
                                   input_grad_ptr,
                                   m.handle.workSpace));
}

} // namespace Pool2D
} // namespace Kernels
} // namespace FlexFlow

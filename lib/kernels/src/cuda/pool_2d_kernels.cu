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

#include "kernels/cuda_helper.h"
#include "kernels/pool_2d_kernels.h"

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
    ffActivationDescriptor_t actiDesc;
    ffPoolingDescriptor_t poolDesc;

    checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
    checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
    checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
    checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));

    checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        input_n,
                                        input_c,
                                        input_h,
                                        input_w));
    cudnnPoolingMode_t mode;
    if (pool_type == PoolOp::MAX) {
      mode = CUDNN_POOLING_MAX;
    } else {
      assert(pool_type == PoolOp::AVG);
      mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    }

    checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc,
                                         mode,
                                         CUDNN_PROPAGATE_NAN,
                                         kernel_h,
                                         kernel_w,
                                         pad_h,
                                         pad_w,
                                         stride_h,
                                         stride_w));
    
  int n, c, h, w;
  checkCUDNN(cudnnGetPooling2dForwardOutputDim(poolDesc, inputTensor, &n, &c, &h, &w));
  assert(n == output_n);
  assert(c == output_c);
  assert(h == output_h);
  assert(w == output_w);

  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  bool relu = false;
  if(activation == Activation::RELU) {
    relu = true;
  }
  Pool2DPerDeviceState state = {handle, inputTensor, outputTensor, actiDesc, poolDesc, relu};
  return state;
} 

void init_kernel(Pool2DPerDeviceState *m,
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
                 PoolType pool_type) {
  checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        input_n,
                                        input_c,
                                        input_h,
                                        input_w));

  cudnnPoolingMode_t mode;
  if (pool_type == POOL_MAX) {
    mode = CUDNN_POOLING_MAX;
  } else {
    assert(pool_type == POOL_AVG);
    mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  }
  checkCUDNN(cudnnSetPooling2dDescriptor(m->poolDesc,
                                         mode,
                                         CUDNN_PROPAGATE_NAN,
                                         kernel_h,
                                         kernel_w,
                                         pad_h,
                                         pad_w,
                                         stride_h,
                                         stride_w));
  int n, c, h, w;
  checkCUDNN(cudnnGetPooling2dForwardOutputDim(
      m->poolDesc, m->inputTensor, &n, &c, &h, &w));
  assert(n == output_n);
  assert(c == output_c);
  assert(h == output_h);
  assert(w == output_w);

  checkCUDNN(cudnnSetTensor4dDescriptor(
      m->outputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
}

void forward_kernel(cudaStream_t stream,
                    Pool2DPerDeviceState const *m,
                    void const *input_ptr,
                    void *output_ptr) {

  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnPoolingForward(m->handle.dnn,
                                 m->poolDesc,
                                 &alpha,
                                 m->inputTensor,
                                 input_ptr,
                                 &beta,
                                 m->outputTensor,
                                 output_ptr));
}

void backward_kernel(cudaStream_t stream,
                     Pool2DPerDeviceState const *m,
                     void const *input_ptr,
                     void *input_grad_ptr,
                     void const *output_ptr,
                     void const *output_grad_ptr) {

  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f;
  checkCUDNN(cudnnPoolingBackward(m->handle.dnn,
                                  m->poolDesc,
                                  &alpha,
                                  m->outputTensor,
                                  output_ptr,
                                  m->outputTensor,
                                  output_grad_ptr,
                                  m->inputTensor,
                                  input_ptr,
                                  &alpha,
                                  m->inputTensor,
                                  input_grad_ptr));
}

} // namespace Pool2D
} // namespace Kernels
} // namespace FlexFlow

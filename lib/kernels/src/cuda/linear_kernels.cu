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

#include "kernels/allocation.h"
#include "kernels/cuda_helper.h"
#include "kernels/linear_kernels.h"

namespace FlexFlow {

namespace Kernels {
namespace Linear {

// what's the float * one_ptr
LinearPerDeviceState
    init_kernel(PerDeviceFFHandle handle, Allocator allocator, float *one_ptr;
                ActiMode activation,
                Regularizer regularizer,
                bool use_bias,
                DataType input_type,
                DataType weight_type,
                DataType output_type,
                int batch_size,
                int channel) {
  ffTensorDescriptor_t outputTensor;
  ffActivationDescriptor_t actiDesc;
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        ff_to_cudnn_datatype(output_type),
                                        batch_size,
                                        channel,
                                        1,
                                        1));
  cudnnActivationMode_t mode;
  switch (activation) {
    case RELU:
      mode = CUDNN_ACTIVATION_RELU;
      break;
    case SIGMOID:
      mode = CUDNN_ACTIVATION_SIGMOID;
      break;
    case TANH:
      mode = CUDNN_ACTIVATION_TANH;
      break;
    case GELU:
      mode = CUDNN_ACTIVATION_GELU;
      break;
    default:
      // Unsupported activation mode
      assert(false);
  }
  checkCUDNN(
      cudnnSetActivationDescriptor(actiDesc, mode, CUDNN_PROPAGATE_NAN, 0.0));
  checkCUDNN(
      cudnnSetTensorDescriptorFromArrayShape(outputTensor, output_shape));

  // todo: how to use allocator to allocate memory for float * one_ptr, how many
  // bytes to allocate?
  checkCUDA(cudaMalloc(one_ptr, sizeof(float) * batch_size));
  LinearPerDeviceState per_device_state = {handle,
                                           outputTensor,
                                           actiDesc,
                                           one_ptr,
                                           activation,
                                           regularizer,
                                           use_bias,
                                           input_type,
                                           weight_type,
                                           output_type};
  return per_device_state;
}

void forward_kernel(cudaStream_t stream,
                    LinearPerDeviceState const &m,
                    void const *input_ptr,
                    void *output_ptr,
                    void const *weight_ptr,
                    void const *bias_ptr,
                    int in_dim,
                    int out_dim,
                    int batch_size) {

  checkCUDA(cublasSetStream(m.handle.blas, stream));
  checkCUDNN(cudnnSetStream(m.handle.dnn, stream));
  float alpha = 1.0f, beta = 0.0f;
  cudaDataType_t input_type = ff_to_cuda_datatype(m.input_type);
  cudaDataType_t weight_type = ff_to_cuda_datatype(m.weight_type);
  cudaDataType_t output_type = ff_to_cuda_datatype(m.output_type);
#if CUDA_VERSION >= 11000
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
  cudaDataType_t compute_type = CUDA_R_32F;
#endif
  checkCUDA(cublasGemmEx(m.handle.blas,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         out_dim,
                         batch_size,
                         in_dim,
                         &alpha,
                         weight_ptr,
                         weight_type,
                         in_dim,
                         input_ptr,
                         input_type,
                         in_dim,
                         &beta,
                         output_ptr,
                         output_type,
                         out_dim,
                         compute_type,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  // use_bias = True
  if (bias_ptr != NULL) {
    checkCUDA(cublasGemmEx(m.handle.blas,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           out_dim,
                           batch_size,
                           1,
                           &alpha,
                           bias_ptr,
                           weight_type,
                           1,
                           m.one_ptr,
                           CUDA_R_32F,
                           1,
                           &alpha,
                           output_ptr,
                           output_type,
                           out_dim,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  if (use_activation(m.activation)) {
    checkCUDNN(cudnnActivationForward(m.handle.dnn,
                                      m.actiDesc,
                                      &alpha,
                                      m.outputTensor,
                                      output_ptr,
                                      &beta,
                                      m.outputTensor,
                                      output_ptr));
  } else if (m.activation == AC_MODE_GELU) {
    size_t elements = (size_t)out_dim * (size_t)batch_size;
    constexpr float B = 0.7978845608028654f;   // sqrt(2.0/M_PI)
    constexpr float C = 0.035677408136300125f; // 0.044715 * sqrt(2.0/M_PI)
    gelu_forward_kernel<<<GET_BLOCKS(elements), CUDA_NUM_THREADS>>>(
        elements, B, C, (float *)output_ptr);
  } else if (m.activation == AC_MODE_NONE) {
    // Do nothing
  } else {
    assert(false && "Unsupported activation for Linear");
  }
}

void backward_kernel(cudaStream_t stream,
                     LinearPerDeviceState const &m,
                     void const *input_ptr,
                     void *input_grad_ptr,
                     void const *output_ptr,
                     void *output_grad_ptr,
                     void const *kernel_ptr,
                     void *kernel_grad_ptr,
                     void *bias_grad_ptr,
                     int in_dim,
                     int out_dim,
                     int batch_size) {

  checkCUDA(cublasSetStream(m.handle.blas, stream));
  checkCUDNN(cudnnSetStream(m.handle.dnn, stream));

  float alpha = 1.0f;
  cudaDataType_t input_type = ff_to_cuda_datatype(m.input_type);
  cudaDataType_t weight_type = ff_to_cuda_datatype(m.weight_type);
  cudaDataType_t output_type = ff_to_cuda_datatype(m.output_type);
#if CUDA_VERSION >= 11000
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
  cudaDataType_t compute_type = CUDA_R_32F;
#endif
  int output_size = out_dim * batch_size;
  if (m.activation == AC_MODE_RELU) {
    relu_backward_kernel(
        m.output_type, output_grad_ptr, output_ptr, output_size, stream);
  } else if (m.activation == AC_MODE_SIGMOID) {
    sigmoid_backward_kernel(
        m.output_type, output_grad_ptr, output_ptr, output_size, stream);
  } else {
    // TODO: only support relu and sigmoid for now
    assert(m.activation == AC_MODE_NONE);
  }
  // Compute weight gradiant
  // NOTE: we use alpha=1 for kernel_grad to accumulate gradients
  checkCUDA(cublasGemmEx(m.handle.blas,
                         CUBLAS_OP_N,
                         CUBLAS_OP_T,
                         in_dim,
                         out_dim,
                         batch_size,
                         &alpha,
                         input_ptr,
                         input_type,
                         in_dim,
                         output_grad_ptr,
                         output_type,
                         out_dim,
                         &alpha,
                         kernel_grad_ptr,
                         weight_type,
                         in_dim,
                         compute_type,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  if (m.kernel_reg_type == REG_MODE_NONE) {
    // do nothing
  } else if (m.kernel_reg_type == REG_MODE_L2) {
    checkCUDA(cublasSgeam(m.handle.blas,
                          CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          in_dim,
                          out_dim,
                          &alpha,
                          (float *)kernel_grad_ptr,
                          in_dim,
                          &(m.kernel_reg_lambda),
                          (float *)kernel_ptr,
                          in_dim,
                          (float *)kernel_grad_ptr,
                          in_dim));
  } else {
    assert(false && "Only L2 regularization is supported");
  }

  // Compute bias gradiant
  // NOTE: we use alpha=1 for bias_grad to accumulate gradients
  // use_bias = True
  if (bias_grad_ptr != NULL) {
    checkCUDA(cublasGemmEx(m.handle.blas,
                           CUBLAS_OP_N,
                           CUBLAS_OP_T,
                           1,
                           out_dim,
                           batch_size,
                           &alpha,
                           m.one_ptr,
                           CUDA_R_32F,
                           1,
                           output_grad_ptr,
                           output_type,
                           out_dim,
                           &alpha,
                           bias_grad_ptr,
                           weight_type,
                           1,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  // Compute data gradiant
  // NOTE: we use alpha=1 for input_grad to accumulate gradients
  if (input_grad_ptr != NULL) {
    checkCUDA(cublasGemmEx(m.handle.blas,
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           in_dim,
                           batch_size,
                           out_dim,
                           &alpha,
                           kernel_ptr,
                           weight_type,
                           in_dim,
                           output_grad_ptr,
                           output_type,
                           out_dim,
                           &alpha,
                           input_grad_ptr,
                           input_type,
                           in_dim,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
}

} // namespace Linear
} // namespace Kernels
} // namespace FlexFlow

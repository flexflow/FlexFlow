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

#include "kernels/linear_kernels.h"
#include "kernels/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

namespace Kernels {
namespace Linear {

bool use_activation(std::optional<Activation> activation) {
  if (activation.has_value()) {
    switch (activation.value()) {
      case Activation::RELU:
      case Activation::SIGMOID:
      case Activation::TANH:
        return true;
      case Activation::GELU:
        return false;
      default:
        assert(false && "Unsupported activation for Linear");
        break;
    }
  }
  return false;
}

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
  checkCUDNN(miopenCreateTensorDescriptor(&outputTensor));
  checkCUDNN(miopenSetActivationDescriptor(actiDesc, mode, 0.0, 0.0, 0.0));
  checkCUDNN(miopenSet4dTensorDescriptor(outputTensor,
                                         ff_to_cudnn_datatype(output_type),
                                         batch_size,
                                         channel,
                                         1,
                                         1));

  miopenActivationMode_t mode;
  if (activation.has_value()) {
    switch (activation.value()) {
      case Activation::RELU:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case Activation::SIGMOID:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      case Activation::TANH:
        mode = CUDNN_ACTIVATION_TANH;
        break;
      case Activation::GELU:
        // mode = CUDNN_ACTIVATION_GELU; //cudnnActivationMode_t does not have
        // GELU
        break;
      default:
        // Unsupported activation mode
        assert(false);
    }
  }
  checkCUDNN(miopenSetActivationDescriptor(actiDesc, mode, 0.0, 0.0, 0.0));
  // todo: how to use allocator to allocate memory for float * one_ptr, how many
  // bytes to allocate?
  checkCUDA(hipMalloc(&one_ptr, sizeof(float) * batch_size));
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

void forward_kernel(hipStream_t stream,
                    LinearPerDeviceState const &m,
                    void const *input_ptr,
                    void *output_ptr,
                    void const *weight_ptr,
                    void const *bias_ptr,
                    int in_dim,
                    int out_dim,
                    int batch_size) {

  checkCUDA(hipblasSetStream(m.handle.blas, stream));
  checkCUDNN(miopenSetStream(m.handle.dnn, stream));
  float alpha = 1.0f, beta = 0.0f;
  hipblasDatatype_t input_type = ff_to_cuda_datatype(m.input_type);
  hipblasDatatype_t weight_type = ff_to_cuda_datatype(m.weight_type);
  hipblasDatatype_t output_type = ff_to_cuda_datatype(m.output_type);
#if CUDA_VERSION >= 11000
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
  hipblasDatatype_t compute_type = HIPBLAS_R_32F;
#endif
  checkCUDA(hipblasGemmEx(m.handle.blas,
                          HIPBLAS_OP_T,
                          HIPBLAS_OP_N,
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
                          HIPBLAS_GEMM_DEFAULT));
  // use_bias = True
  if (bias_ptr != NULL) {
    checkCUDA(hipblasGemmEx(m.handle.blas,
                            HIPBLAS_OP_T,
                            HIPBLAS_OP_N,
                            out_dim,
                            batch_size,
                            1,
                            &alpha,
                            bias_ptr,
                            weight_type,
                            1,
                            m.one_ptr,
                            HIPBLAS_R_32F,
                            1,
                            &alpha,
                            output_ptr,
                            output_type,
                            out_dim,
                            compute_type,
                            HIPBLAS_GEMM_DEFAULT));
  }
  if (use_activation(m.activation)) {
    checkCUDNN(miopenActivationForward(m.handle.dnn,
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
    hipLaunchKernelGGL(gelu_forward_kernel,
                       GET_BLOCKS(elements),
                       CUDA_NUM_THREADS,
                       0,
                       0,
                       elements,
                       B,
                       C,
                       (float *)output_ptr);
  } else {
    // Do nothing
  }
}

void backward_kernel(hipStream_t stream,
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

  checkCUDA(hipblasSetStream(m.handle.blas, stream));
  checkCUDNN(miopenSetStream(m.handle.dnn, stream));

  float alpha = 1.0f;
  hipblasDatatype_t input_type = ff_to_cuda_datatype(m.input_type);
  hipblasDatatype_t weight_type = ff_to_cuda_datatype(m.weight_type);
  hipblasDatatype_t output_type = ff_to_cuda_datatype(m.output_type);
#if CUDA_VERSION >= 11000
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
  hipblasDatatype_t compute_type = HIPBLAS_R_32F;
#endif
  int output_size = out_dim * batch_size;
  if (m.activation.has_value()) {
    if (m.activation == Activation::RELU) {
      relu_backward_kernel(
          m.output_type, output_grad_ptr, output_ptr, output_size, stream);
    } else if (m.activation == Activation::SIGMOID) {
      sigmoid_backward_kernel(
          m.output_type, output_grad_ptr, output_ptr, output_size, stream);
    } else {
      // TODO: only support relu and sigmoid for now
      assert(false && "Unsupported activation for Linear");
    }
  }
  // Compute weight gradiant
  // NOTE: we use alpha=1 for kernel_grad to accumulate gradients
  checkCUDA(hipblasGemmEx(m.handle.blas,
                          HIPBLAS_OP_N,
                          HIPBLAS_OP_T,
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
                          HIPBLAS_GEMM_DEFAULT));

  if (m.regularizer == std::nullopt) {
    // do nothing
  } else {
    RegularizerAttrs regularizer_attrs = m.regularizer.value();
    if (std::holds_alternative<L2RegularizerAttrs>(regularizer_attrs)) {
      L2RegularizerAttrs l2_attrs =
          std::get<L2RegularizerAttrs>(regularizer_attrs);
      float lambda = l2_attrs.lambda;
      checkCUDA(hipblasSgeam(m.handle.blas,
                             HIPBLAS_OP_N,
                             HIPBLAS_OP_N,
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
  }
  // compute bias gradient
  // NOTE: we use alpha=1 for bias_grad to accumulate gradients
  // use_bias = True
  if (bias_grad_ptr != NULL) {
    checkCUDA(hipblasGemmEx(m.handle.blas,
                            HIPBLAS_OP_N,
                            HIPBLAS_OP_T,
                            1,
                            out_dim,
                            batch_size,
                            &alpha,
                            m.one_ptr,
                            HIPBLAS_R_32F,
                            1,
                            output_grad_ptr,
                            output_type,
                            out_dim,
                            &alpha,
                            bias_grad_ptr,
                            weight_type,
                            1,
                            compute_type,
                            HIPBLAS_GEMM_DEFAULT));
  }
  // Compute data gradiant
  // NOTE: we use alpha=1 for input_grad to accumulate gradients
  if (input_grad_ptr != NULL) {
    checkCUDA(hipblasGemmEx(m.handle.blas,
                            HIPBLAS_OP_N,
                            HIPBLAS_OP_N,
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
                            HIPBLAS_GEMM_DEFAULT));
  }
}

} // namespace Linear
} // namespace Kernels
} // namespace FlexFlow

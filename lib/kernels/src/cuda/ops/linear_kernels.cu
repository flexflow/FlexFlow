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
#include "kernels/allocation.h"
#include "kernels/linear_kernels_gpu.h"
#include "utils/integer_conversions.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

namespace Kernels {
namespace Linear {

static bool use_activation(std::optional<Activation> activation) {
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
    gpu_init_kernel(PerDeviceFFHandle handle,
                    std::optional<Activation> activation,
                    std::optional<RegularizerAttrs> regularizer,
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
  } else {
    mode = CUDNN_ACTIVATION_IDENTITY;
  }
  checkCUDNN(
      cudnnSetActivationDescriptor(actiDesc, mode, CUDNN_PROPAGATE_NAN, 0.0));
  // don't need this line below because we are already setting 4dDescriptor for
  // outputTensor above checkCUDNN(
  //     cudnnSetTensorDescriptorFromArrayShape(outputTensor, output_shape));

  // todo: how to use allocator to allocate memory for float * one_ptr, how many
  // bytes to allocate?
  float *one_ptr;
  checkCUDA(cudaMalloc(&one_ptr, sizeof(float) * batch_size));
  float one_ptr_cpu[batch_size];
  for (int i = 0; i < batch_size; i++) {
    one_ptr_cpu[i] = 1.0;
  }
  checkCUDA(cudaMemcpy(one_ptr,
                       one_ptr_cpu,
                       sizeof(float) * batch_size,
                       cudaMemcpyHostToDevice));
  LinearPerDeviceState per_device_state = LinearPerDeviceState{
      /*handle=*/handle,
      /*outputTensor=*/outputTensor,
      /*actiDesc=*/actiDesc,
      /*one_ptr=*/one_ptr,
      /*mode=*/mode,
      /*activation=*/activation,
      /*regularizer=*/regularizer,
      /*use_bias=*/use_bias,
      /*input_type=*/input_type,
      /*weight_type=*/weight_type,
      /*output_type=*/output_type,
  };
  return per_device_state;
}

void gpu_forward_kernel(cudaStream_t stream,
                        LinearPerDeviceState const &m,
                        float const *input_ptr,
                        float *output_ptr,
                        float const *weight_ptr,
                        float const *bias_ptr,
                        int in_dim,
                        int out_dim,
                        int batch_size) {

  checkCUBLAS(cublasSetStream(m.handle.blas, stream));
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
  checkCUBLAS(cublasGemmEx(m.handle.blas,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           out_dim,
                           batch_size,
                           in_dim,
                           &alpha,
                           static_cast<void const *>(weight_ptr),
                           weight_type,
                           in_dim,
                           static_cast<void const *>(input_ptr),
                           input_type,
                           in_dim,
                           &beta,
                           static_cast<void *>(output_ptr),
                           output_type,
                           out_dim,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  if (bias_ptr != nullptr) {
    checkCUBLAS(cublasGemmEx(m.handle.blas,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             out_dim,
                             batch_size,
                             1,
                             &alpha,
                             static_cast<void const *>(bias_ptr),
                             weight_type,
                             out_dim,
                             static_cast<void const *>(m.one_ptr),
                             CUDA_R_32F,
                             1,
                             &alpha,
                             static_cast<void *>(output_ptr),
                             output_type,
                             out_dim,
                             compute_type,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  // if (use_activation(m.activation)) {
  //   checkCUDNN(cudnnActivationForward(m.handle.dnn,
  //                                     m.actiDesc,
  //                                     &alpha,
  //                                     m.outputTensor,
  //                                     static_cast<void *>(output_ptr),
  //                                     &beta,
  //                                     m.outputTensor,
  //                                     static_cast<void *>(output_ptr)));
  // } else if (m.activation == Activation::GELU) {
  //   size_t elements = size_t_from_int(out_dim) * size_t_from_int(batch_size);
  //   constexpr float B = 0.7978845608028654f;   // sqrt(2.0/M_PI)
  //   constexpr float C = 0.035677408136300125f; // 0.044715 * sqrt(2.0/M_PI)
  //   gelu_forward_kernel<<<GET_BLOCKS(elements), CUDA_NUM_THREADS>>>(
  //       elements, B, C, (float *)output_ptr);
  // } else {
  //   // Do nothing
  // }
}

void gpu_backward_kernel(cudaStream_t stream,
                         LinearPerDeviceState const &m,
                         float const *output_ptr,
                         float *output_grad_ptr,
                         float const *input_ptr,
                         float *input_grad_ptr,
                         float const *kernel_ptr,
                         float *kernel_grad_ptr,
                         float *bias_grad_ptr,
                         int in_dim,
                         int out_dim,
                         int batch_size) {
  checkCUBLAS(cublasSetStream(m.handle.blas, stream));
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
  if (m.activation.has_value()) {
    if (m.activation == Activation::RELU) {
      relu_backward_kernel(m.output_type,
                           static_cast<void *>(output_grad_ptr),
                           static_cast<void const *>(output_ptr),
                           output_size,
                           stream);
    } else if (m.activation == Activation::SIGMOID) {
      sigmoid_backward_kernel(m.output_type,
                              static_cast<void *>(output_grad_ptr),
                              static_cast<void const *>(output_ptr),
                              output_size,
                              stream);
    } else {
      // TODO: only support relu and sigmoid for now
      PANIC("Unsupported activation for Linear", m.activation.value());
    }
  }

  // Compute weight gradiant
  // NOTE: we use alpha=1 for kernel_grad to accumulate gradients
  checkCUBLAS(cublasGemmEx(m.handle.blas,
                           CUBLAS_OP_N,
                           CUBLAS_OP_T,
                           in_dim,
                           out_dim,
                           batch_size,
                           &alpha,
                           static_cast<void const *>(input_ptr),
                           input_type,
                           in_dim,
                           static_cast<void *>(output_grad_ptr),
                           output_type,
                           out_dim,
                           &alpha,
                           static_cast<void *>(kernel_grad_ptr),
                           weight_type,
                           in_dim,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  if (m.regularizer == std::nullopt) {
    // do nothing
  } else {
    RegularizerAttrs regularizer_attrs = m.regularizer.value();
    if (regularizer_attrs.has<L2RegularizerAttrs>()) {
      L2RegularizerAttrs l2_attrs = regularizer_attrs.get<L2RegularizerAttrs>();
      float lambda = l2_attrs.lambda;
      checkCUBLAS(cublasSgeam(m.handle.blas,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              in_dim,
                              out_dim,
                              &alpha,
                              kernel_grad_ptr,
                              in_dim,
                              &lambda,
                              kernel_ptr,
                              in_dim,
                              kernel_grad_ptr,
                              in_dim));
    } else {
      assert(false && "Only L2 regularization is supported");
    }
  }

  // Compute bias gradiant
  // NOTE: we use alpha=1 for bias_grad to accumulate gradients
  // use_bias = True
  if (bias_grad_ptr != NULL) {
    checkCUBLAS(cublasGemmEx(m.handle.blas,
                             CUBLAS_OP_N,
                             CUBLAS_OP_T,
                             1,
                             out_dim,
                             batch_size,
                             &alpha,
                             static_cast<void const *>(m.one_ptr),
                             CUDA_R_32F,
                             1,
                             static_cast<void *>(output_grad_ptr),
                             output_type,
                             out_dim,
                             &alpha,
                             static_cast<void *>(bias_grad_ptr),
                             weight_type,
                             1,
                             compute_type,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  // Compute data gradiant
  // NOTE: we use alpha=1 for input_grad to accumulate gradients
  if (input_grad_ptr != NULL) {
    checkCUBLAS(cublasGemmEx(m.handle.blas,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             in_dim,
                             batch_size,
                             out_dim,
                             &alpha,
                             static_cast<void const *>(kernel_ptr),
                             weight_type,
                             in_dim,
                             static_cast<void *>(output_grad_ptr),
                             output_type,
                             out_dim,
                             &alpha,
                             static_cast<void *>(input_grad_ptr),
                             input_type,
                             in_dim,
                             compute_type,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
}

void gpu_cleanup_kernel(LinearPerDeviceState &per_device_state) {
  NOT_IMPLEMENTED();
}

} // namespace Linear
} // namespace Kernels
} // namespace FlexFlow

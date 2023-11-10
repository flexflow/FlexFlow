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

#include "flexflow/ops/kernels/linear_kernels.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

LinearMeta::LinearMeta(FFHandler handler,
                       int batch_size,
                       Linear const *li,
                       MemoryAllocator gpu_mem_allocator,
                       int weightSize)
    : OpMeta(handler, li) {
  // Allocate an all-one's vector
  float *dram_one_ptr = (float *)malloc(sizeof(float) * batch_size);
  for (int i = 0; i < batch_size; i++) {
    dram_one_ptr[i] = 1.0f;
  }
  float *fb_one_ptr;
  checkCUDA(hipMalloc(&fb_one_ptr, sizeof(float) * batch_size));
  checkCUDA(hipMemcpy(fb_one_ptr,
                      dram_one_ptr,
                      sizeof(float) * batch_size,
                      hipMemcpyHostToDevice));
  one_ptr = (void *)fb_one_ptr;
  // Allocate descriptors
  checkCUDNN(miopenCreateActivationDescriptor(&actiDesc));
  checkCUDNN(miopenCreateTensorDescriptor(&outputTensor));
}
LinearMeta::~LinearMeta(void) {}

namespace Kernels {
namespace Linear {

bool use_activation(ActiMode mode) {
  switch (mode) {
    case AC_MODE_RELU:
    case AC_MODE_SIGMOID:
    case AC_MODE_TANH:
      return true;
    case AC_MODE_NONE:
      return false;
    default:
      assert(0);
      break;
  }
  return false;
}

void init_kernel(LinearMeta *m, int batch_size, int channel) {
  if (use_activation(m->activation)) {
    miopenActivationMode_t mode;
    switch (m->activation) {
      case AC_MODE_RELU:
        mode = miopenActivationRELU;
        break;
      case AC_MODE_SIGMOID:
        mode = miopenActivationLOGISTIC;
        break;
      default:
        // Unsupported activation mode
        assert(false);
    }
    checkCUDNN(miopenSetActivationDescriptor(m->actiDesc, mode, 0.0, 0.0, 0.0));
    checkCUDNN(
        miopenSet4dTensorDescriptor(m->outputTensor,
                                    ff_to_cudnn_datatype(m->output_type[0]),
                                    batch_size,
                                    channel,
                                    1,
                                    1));
  }
}

void forward_kernel_wrapper(LinearMeta const *m,
                            void const *input_ptr,
                            void *output_ptr,
                            void const *weight_ptr,
                            void const *bias_ptr,
                            int in_dim,
                            int out_dim,
                            int batch_size) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
  }

  if (m->input_type[0] == DT_FLOAT) {
    Internal::forward_kernel<float>(m,
                                    input_ptr,
                                    output_ptr,
                                    weight_ptr,
                                    bias_ptr,
                                    in_dim,
                                    out_dim,
                                    batch_size,
                                    stream);
  } else if (m->input_type[0] == DT_HALF) {
    Internal::forward_kernel<half>(m,
                                   input_ptr,
                                   output_ptr,
                                   weight_ptr,
                                   bias_ptr,
                                   in_dim,
                                   out_dim,
                                   batch_size,
                                   stream);
  }

  if (m->profiling) {
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
    printf("%s [Linear] forward time = %.2lfms\n", m->op_name, elapsed);
    // print_tensor<float>(acc_input.ptr, acc_input.rect.volume(),
    // "[Linear:forward:input]"); print_tensor<float>(acc_kernel.ptr,
    // acc_kernel.rect.volume(), "[Linear:forward:kernel]");
    // print_tensor<float>(acc_bias.ptr, acc_bias.rect.volume(),
    // "[Linear:forward:bias]"); print_tensor<float>(acc_output.ptr,
    // acc_output.rect.volume(), "[Linear:forward:output]");
  }
}

void backward_kernel_wrapper(LinearMeta const *m,
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
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
  }
  if (m->input_type[0] == DT_FLOAT) {
    Internal::backward_kernel<float>(m,
                                     input_ptr,
                                     input_grad_ptr,
                                     output_ptr,
                                     output_grad_ptr,
                                     kernel_ptr,
                                     kernel_grad_ptr,
                                     bias_grad_ptr,
                                     in_dim,
                                     out_dim,
                                     batch_size,
                                     stream);
  } else if (m->input_type[0] == DT_HALF) {
    Internal::backward_kernel<half>(m,
                                    input_ptr,
                                    input_grad_ptr,
                                    output_ptr,
                                    output_grad_ptr,
                                    kernel_ptr,
                                    kernel_grad_ptr,
                                    bias_grad_ptr,
                                    in_dim,
                                    out_dim,
                                    batch_size,
                                    stream);
  }

  if (m->profiling) {
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
    printf("%s Linear backward time = %.2lfms\n", m->op_name, elapsed);
    // print_tensor<float>(acc_output_grad.ptr, acc_output_grad.rect.volume(),
    // "[Linear:backward:output_grad]");
    // print_tensor<float>(acc_kernel_grad.ptr, acc_kernel_grad.rect.volume(),
    // "[Linear:backward:kernel_grad]"); print_tensor<1,
    // float>(acc_bias_grad.ptr, acc_bias_grad.rect,
    // "[Linear:backward:bias_grad]"); print_tensor<float>(input_grad,
    // acc_input.rect.volume(), "[Linear:backward:input_grad]");
  }
}

/*
__host__
Parameter* Linear::get_parameter(int index)
{
  if (index == 0) {
    return &weights[0];
  } else if (index == 1){
    return &weights[1];
  } else {
    assert(0);
    return NULL;
  }
}
*/

namespace Internal {
template <typename DT>
void forward_kernel(LinearMeta const *m,
                    void const *input_ptr,
                    void *output_ptr,
                    void const *weight_ptr,
                    void const *bias_ptr,
                    int in_dim,
                    int out_dim,
                    int batch_size,
                    hipStream_t stream) {
  checkCUDA(hipblasSetStream(m->handle.blas, stream));
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));
  DT alpha = 1.0f, beta = 0.0f;
  hipblasDatatype_t input_type = ff_to_cuda_datatype(m->input_type[0]);
  hipblasDatatype_t weight_type = ff_to_cuda_datatype(m->weight_type[0]);
  hipblasDatatype_t output_type = ff_to_cuda_datatype(m->output_type[0]);
#if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
  hipblasDatatype_t compute_type = output_type;
#else
  // TODO: currently use the output_type
  // cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
  hipblasDatatype_t compute_type = output_type;
#endif
  checkCUDA(hipblasGemmEx(m->handle.blas,
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
    checkCUDA(hipblasGemmEx(m->handle.blas,
                            HIPBLAS_OP_T,
                            HIPBLAS_OP_N,
                            out_dim,
                            batch_size,
                            1,
                            &alpha,
                            bias_ptr,
                            weight_type,
                            1,
                            static_cast<DT *>(m->one_ptr),
                            weight_type,
                            1,
                            &alpha,
                            output_ptr,
                            output_type,
                            out_dim,
                            compute_type,
                            HIPBLAS_GEMM_DEFAULT));
  }
  if (use_activation(m->activation)) {
    checkCUDNN(miopenActivationForward(m->handle.dnn,
                                       m->actiDesc,
                                       &alpha,
                                       m->outputTensor,
                                       output_ptr,
                                       &beta,
                                       m->outputTensor,
                                       output_ptr));
  } else if (m->activation == AC_MODE_GELU) {
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
  } else if (m->activation == AC_MODE_NONE) {
    // Do nothing
  } else {
    assert(false && "Unsupported activation for Linear");
  }
}

template <typename DT>
void backward_kernel(LinearMeta const *m,
                     void const *input_ptr,
                     void *input_grad_ptr,
                     void const *output_ptr,
                     void *output_grad_ptr,
                     void const *kernel_ptr,
                     void *kernel_grad_ptr,
                     void *bias_grad_ptr,
                     int in_dim,
                     int out_dim,
                     int batch_size,
                     hipStream_t stream) {
  checkCUDA(hipblasSetStream(m->handle.blas, stream));
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

  DT alpha = 1.0f;
  hipblasDatatype_t input_type = ff_to_cuda_datatype(m->input_type[0]);
  hipblasDatatype_t weight_type = ff_to_cuda_datatype(m->weight_type[0]);
  hipblasDatatype_t output_type = ff_to_cuda_datatype(m->output_type[0]);
#if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
  hipblasDatatype_t compute_type = output_type;
#else
  // TODO: currently use output_type
  // cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
  hipblasDatatype_t compute_type = output_type;
#endif
  int output_size = out_dim * batch_size;
  if (m->activation == AC_MODE_RELU) {
    relu_backward_kernel(
        m->output_type[0], output_grad_ptr, output_ptr, output_size, stream);
  } else if (m->activation == AC_MODE_SIGMOID) {
    sigmoid_backward_kernel(
        m->output_type[0], output_grad_ptr, output_ptr, output_size, stream);
  } else {
    // TODO: only support relu and sigmoid for now
    assert(m->activation == AC_MODE_NONE);
  }
  // Compute weight gradiant
  // NOTE: we use alpha=1 for kernel_grad to accumulate gradients
  checkCUDA(hipblasGemmEx(m->handle.blas,
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
  // Compute bias gradiant
  // NOTE: we use alpha=1 for bias_grad to accumulate gradients
  // use_bias = True
  if (bias_grad_ptr != NULL) {
    checkCUDA(hipblasGemmEx(m->handle.blas,
                            HIPBLAS_OP_N,
                            HIPBLAS_OP_T,
                            1,
                            out_dim,
                            batch_size,
                            &alpha,
                            m->one_ptr,
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
    checkCUDA(hipblasGemmEx(m->handle.blas,
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

} // namespace Internal
} // namespace Linear
} // namespace Kernels
}; // namespace FlexFlow

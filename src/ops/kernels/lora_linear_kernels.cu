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

#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/kernels/decompress_kernels.h"
#include "flexflow/ops/kernels/lora_linear_kernels.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

LoraLinearMeta::LoraLinearMeta(FFHandler handler, LoraLinear const *li)
    : OpMeta(handler, li) {}

LoraLinearMeta::~LoraLinearMeta(void) {}

namespace Kernels {
namespace LoraLinear {

void inference_kernel_wrapper(LoraLinearMeta *m,
                              void const *input_ptr,
                              void *output_ptr,
                              void const *weight_first_ptr,
                              void const *weight_second_ptr,
                              int in_dim,
                              int out_dim,
                              int rank,
                              int num_infr_tokens,
                              int num_peft_tokens) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  if (m->input_type[0] == DT_FLOAT) {
    Internal::inference_kernel<float>(m,
                                      input_ptr,
                                      output_ptr,
                                      weight_first_ptr,
                                      weight_second_ptr,
                                      in_dim,
                                      out_dim,
                                      rank,
                                      num_infr_tokens,
                                      num_peft_tokens,
                                      stream);
  } else if (m->input_type[0] == DT_HALF) {
    Internal::inference_kernel<half>(m,
                                     input_ptr,
                                     output_ptr,
                                     weight_first_ptr,
                                     weight_second_ptr,
                                     in_dim,
                                     out_dim,
                                     rank,
                                     num_infr_tokens,
                                     num_peft_tokens,
                                     stream);
  }

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("%s [LoraLinear] forward time = %.2lfms\n", m->op_name, elapsed);
    // print_tensor<float>((float*)input_ptr, in_dim * batch_size,
    // "[LoraLinear:forward:input]"); print_tensor<float>((float*)weight_ptr,
    // in_dim
    // * out_dim, "[LoraLinear:forward:kernel]");
    // print_tensor<float>((float*)output_ptr, out_dim * batch_size,
    // "[LoraLinear:forward:output]");
  }
}

void peft_bwd_kernel_wrapper(LoraLinearMeta *m,
                             void *input_grad_ptr,
                             void const *output_grad_ptr,
                             void const *weight_first_ptr,
                             void const *weight_second_ptr,
                             void *weight_first_grad_ptr,
                             void *weight_second_grad_ptr,
                             int in_dim,
                             int out_dim,
                             int rank,
                             int num_infr_tokens,
                             int num_peft_tokens) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  if (m->input_type[0] == DT_FLOAT) {
    Internal::peft_bwd_kernel<float>(m,
                                     input_grad_ptr,
                                     output_grad_ptr,
                                     weight_first_ptr,
                                     weight_second_ptr,
                                     weight_first_grad_ptr,
                                     weight_second_grad_ptr,
                                     in_dim,
                                     out_dim,
                                     rank,
                                     num_infr_tokens,
                                     num_peft_tokens,
                                     stream);
  } else if (m->input_type[0] == DT_HALF) {
    Internal::peft_bwd_kernel<half>(m,
                                    input_grad_ptr,
                                    output_grad_ptr,
                                    weight_first_ptr,
                                    weight_second_ptr,
                                    weight_first_grad_ptr,
                                    weight_second_grad_ptr,
                                    in_dim,
                                    out_dim,
                                    rank,
                                    num_infr_tokens,
                                    num_peft_tokens,
                                    stream);
  }

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("%s [LoraLinear] PEFT Bwd time = %.2lfms\n", m->op_name, elapsed);
    // print_tensor<float>((float*)input_ptr, in_dim * batch_size,
    // "[LoraLinear:forward:input]"); print_tensor<float>((float*)weight_ptr,
    // in_dim
    // * out_dim, "[LoraLinear:forward:kernel]");
    // print_tensor<float>((float*)output_ptr, out_dim * batch_size,
    // "[LoraLinear:forward:output]");
  }
}

namespace Internal {

template <typename DT>
void inference_kernel(LoraLinearMeta *m,
                      void const *input_ptr,
                      void *output_ptr,
                      void const *weight_first_ptr,
                      void const *weight_second_ptr,
                      int in_dim,
                      int out_dim,
                      int rank,
                      int num_infr_tokens,
                      int num_peft_tokens,
                      ffStream_t stream) {
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  DT alpha = 1.0f, beta = 0.0f;
  cudaDataType_t input_type = ff_to_cuda_datatype(m->input_type[0]);
  cudaDataType_t weight_type = ff_to_cuda_datatype(m->weight_type[0]);
  assert(m->weight_type[1] == m->weight_type[0]);
  cudaDataType_t output_type = ff_to_cuda_datatype(m->input_type[1]);
  cudaDataType_t lr_actv_type = output_type;
  assert(input_type == weight_type && weight_type == output_type);
  // adjust input_ptr and output_ptr offset
  // TODO: we currently assume that all inference tokens do not use LoRA
  input_ptr = static_cast<DT const *>(input_ptr) + num_infr_tokens * in_dim;
  output_ptr = static_cast<DT *>(output_ptr) + num_infr_tokens * out_dim;

#if CUDA_VERSION >= 11000
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
  cudaDataType_t compute_type = input_type;
#endif
  MemoryAllocator *allocator = m->handle.peft_activation_allocator;
  m->input_activation = allocator->allocate_instance_untyped(
      data_type_size(m->input_type[0]) * num_peft_tokens * in_dim);
  m->low_rank_activation = allocator->allocate_instance_untyped(
      data_type_size(m->input_type[1]) * num_peft_tokens * rank);
  // copy input activation
  checkCUDA(cudaMemcpyAsync(m->input_activation,
                            input_ptr,
                            data_type_size(m->input_type[0]) * num_peft_tokens *
                                in_dim,
                            cudaMemcpyDeviceToDevice,
                            stream));
  // buffer = weight_first * input
  checkCUDA(cublasGemmEx(m->handle.blas,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         rank,
                         num_peft_tokens,
                         in_dim,
                         &alpha,
                         weight_first_ptr,
                         weight_type,
                         in_dim,
                         input_ptr,
                         input_type,
                         in_dim,
                         &beta,
                         m->low_rank_activation,
                         lr_actv_type,
                         rank,
                         compute_type,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  // output = weight_second * buffer
  // Note that we use alpha in both places since we do
  // an in-place update for LoraLinear
  checkCUDA(cublasGemmEx(m->handle.blas,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         out_dim,
                         num_peft_tokens,
                         rank,
                         &alpha,
                         weight_second_ptr,
                         weight_type,
                         rank,
                         m->low_rank_activation,
                         lr_actv_type,
                         rank,
                         &alpha,
                         output_ptr,
                         output_type,
                         out_dim,
                         compute_type,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template <typename DT>
void peft_bwd_kernel(LoraLinearMeta *m,
                     void *input_grad_ptr,
                     void const *output_grad_ptr,
                     void const *weight_first_ptr,
                     void const *weight_second_ptr,
                     void *weight_first_grad_ptr,
                     void *weight_second_grad_ptr,
                     int in_dim,
                     int out_dim,
                     int rank,
                     int num_infr_tokens,
                     int num_peft_tokens,
                     ffStream_t stream) {
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  DT alpha = 1.0f;
  cudaDataType_t input_type = ff_to_cuda_datatype(m->input_type[0]);
  cudaDataType_t weight_type = ff_to_cuda_datatype(m->weight_type[0]);
  assert(weight_type == ff_to_cuda_datatype(m->weight_type[1]));
  cudaDataType_t output_type = ff_to_cuda_datatype(m->output_type[0]);
  cudaDataType_t lr_actv_type = output_type;
  // update input_grad_ptr and output_grad_ptr offset
  input_grad_ptr = static_cast<DT *>(input_grad_ptr) + num_infr_tokens * in_dim;
  output_grad_ptr =
      static_cast<DT const *>(output_grad_ptr) + num_infr_tokens * out_dim;
#if CUDA_VERSION >= 11000
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
  cudaDataType_t compute_type = CUDA_R_32F;
#endif
  // Compute weight_second gradiant
  // NOTE: we use alpha=1 for weight_second_grad to accumulate gradients
  checkCUDA(cublasGemmEx(m->handle.blas,
                         CUBLAS_OP_N,
                         CUBLAS_OP_T,
                         rank,
                         out_dim,
                         num_peft_tokens,
                         &alpha,
                         m->low_rank_activation,
                         lr_actv_type,
                         rank,
                         output_grad_ptr,
                         output_type,
                         out_dim,
                         &alpha,
                         weight_second_grad_ptr,
                         weight_type,
                         rank,
                         compute_type,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  // Compute gradiants w.r.t. low_rank activation
  // and save the results to low_rank_activation
  // NOTE: we use alpha=1 for input_grad to accumulate gradients
  checkCUDA(cublasGemmEx(m->handle.blas,
                         CUBLAS_OP_N,
                         CUBLAS_OP_N,
                         rank,
                         num_peft_tokens,
                         out_dim,
                         &alpha,
                         weight_second_ptr,
                         weight_type,
                         rank,
                         output_grad_ptr,
                         output_type,
                         out_dim,
                         &alpha,
                         m->low_rank_activation,
                         lr_actv_type,
                         rank,
                         compute_type,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  // Compute weight_first gradiant
  // NOTE: we use alpha=1 for kernel_grad to accumulate gradients
  checkCUDA(cublasGemmEx(m->handle.blas,
                         CUBLAS_OP_N,
                         CUBLAS_OP_T,
                         in_dim,
                         rank,
                         num_peft_tokens,
                         &alpha,
                         m->input_activation,
                         input_type,
                         in_dim,
                         m->low_rank_activation,
                         lr_actv_type,
                         rank,
                         &alpha,
                         weight_first_grad_ptr,
                         weight_type,
                         in_dim,
                         compute_type,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  // Compute input gradiant
  // NOTE: we use alpha=1 for input_grad to accumulate gradients
  if (input_grad_ptr != nullptr) {
    checkCUDA(cublasGemmEx(m->handle.blas,
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           in_dim,
                           num_peft_tokens,
                           rank,
                           &alpha,
                           weight_first_ptr,
                           weight_type,
                           in_dim,
                           m->low_rank_activation,
                           lr_actv_type,
                           rank,
                           &alpha,
                           input_grad_ptr,
                           input_type,
                           in_dim,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
}

} // namespace Internal
} // namespace LoraLinear
} // namespace Kernels
} // namespace FlexFlow

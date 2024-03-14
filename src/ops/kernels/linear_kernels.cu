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
#include "flexflow/ops/kernels/linear_kernels.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

LinearMeta::LinearMeta(FFHandler handler,
                       int batch_size,
                       Linear const *li,
                       MemoryAllocator gpu_mem_allocator,
                       int weightSize)
    : OpMeta(handler, li), weight_ptr(nullptr) {
  DataType data_type = li->data_type;
  // allocate weight and bias in the reserve space for cpu offloading
  if (li->offload) {
    weight_ptr = gpu_mem_allocator.allocate_reserved_untyped(
        weightSize * data_type_size(data_type));
    if (li->quantization_type != DT_NONE) {
      quantized_weightSize = get_quantization_to_byte_size(
          data_type, li->quantization_type, weightSize);
      quantized_weight_ptr =
          gpu_mem_allocator.allocate_reserved<char>(quantized_weightSize);
    }
  }
  // Allocate an all-one's vector
  gpu_mem_allocator.create_legion_instance(
      reserveInst, data_type_size(data_type) * batch_size);
  one_ptr = gpu_mem_allocator.allocate_instance_untyped(
      data_type_size(data_type) * batch_size);
  int parallelism = batch_size;
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  if (data_type == DT_FLOAT) {
    Kernels::Linear::Internal::
        build_one_ptr<<<GET_BLOCKS(parallelism),
                        min(CUDA_NUM_THREADS, parallelism),
                        0,
                        stream>>>((float *)one_ptr, batch_size);
  } else if (data_type == DT_HALF) {
    Kernels::Linear::Internal::
        build_one_ptr<<<GET_BLOCKS(parallelism),
                        min(CUDA_NUM_THREADS, parallelism),
                        0,
                        stream>>>((half *)one_ptr, batch_size);
  }

  // Allocate descriptors
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
}

LinearMeta::~LinearMeta(void) {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
}

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
    cudnnActivationMode_t mode;
    switch (m->activation) {
      case AC_MODE_RELU:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case AC_MODE_SIGMOID:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      default:
        // Unsupported activation mode
        assert(false);
    }
    checkCUDNN(cudnnSetActivationDescriptor(
        m->actiDesc, mode, CUDNN_PROPAGATE_NAN, 0.0));
    checkCUDNN(
        cudnnSetTensor4dDescriptor(m->outputTensor,
                                   CUDNN_TENSOR_NCHW,
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
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
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
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("%s [Linear] forward time = %.2lfms\n", m->op_name, elapsed);
    // print_tensor<float>((float*)input_ptr, in_dim * batch_size,
    // "[Linear:forward:input]"); print_tensor<float>((float*)weight_ptr, in_dim
    // * out_dim, "[Linear:forward:kernel]");
    // print_tensor<float>((float*)output_ptr, out_dim * batch_size,
    // "[Linear:forward:output]");
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
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
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
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
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
__global__ void AddBiasWithReLU(DT *output_ptr,
                                DT const *bias_ptr,
                                int out_dim,
                                int batch_size) {
  CUDA_KERNEL_LOOP(i, out_dim * batch_size) {
    int bias_idx = i % out_dim;
    DT value = output_ptr[i] + bias_ptr[bias_idx];
    output_ptr[i] = ((float)value > 0.0f) ? value : (DT)0.0f;
  }
}

template <typename DT>
void forward_kernel(LinearMeta const *m,
                    void const *input_ptr,
                    void *output_ptr,
                    void const *weight_ptr,
                    void const *bias_ptr,
                    int in_dim,
                    int out_dim,
                    int batch_size,
                    ffStream_t stream) {
  // additional processing for uploading weights
  if (m->offload) {
    // Note that we update weight_ptr when uploading weight
    if (m->quantization_type != DT_NONE) {
      cudaMemcpyAsync(m->quantized_weight_ptr,
                      weight_ptr,
                      m->quantized_weightSize,
                      cudaMemcpyHostToDevice,
                      stream);
      if (m->quantization_type == DT_INT4) {
        int parallelism = in_dim * out_dim / 2;
        decompress_int4_general_weights<DT>
            <<<GET_BLOCKS(parallelism),
               min(CUDA_NUM_THREADS, parallelism),
               0,
               stream>>>(m->quantized_weight_ptr,
                         static_cast<DT *>(m->weight_ptr),
                         in_dim,
                         in_dim * out_dim);
      } else {
        assert(m->quantization_type == DT_INT8);
        int parallelism = in_dim * out_dim;
        decompress_int8_general_weights<DT>
            <<<GET_BLOCKS(parallelism),
               min(CUDA_NUM_THREADS, parallelism),
               0,
               stream>>>(m->quantized_weight_ptr,
                         static_cast<DT *>(m->weight_ptr),
                         in_dim,
                         in_dim * out_dim);
      }

    } else {
      cudaMemcpyAsync(m->weight_ptr,
                      weight_ptr,
                      in_dim * out_dim * sizeof(DT),
                      cudaMemcpyHostToDevice,
                      stream);
    }
  }
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  DT alpha = 1.0f, beta = 0.0f;
  cudaDataType_t input_type = ff_to_cuda_datatype(m->input_type[0]);
  cudaDataType_t weight_type = m->offload
                                   ? ff_to_cuda_datatype(m->weight_ptr_type)
                                   : ff_to_cuda_datatype(m->weight_type[0]);
  cudaDataType_t output_type = ff_to_cuda_datatype(m->output_type[0]);
  assert(input_type == weight_type && weight_type == output_type);
#if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
  cudaDataType_t compute_type = cublas_data_type;
#else
  // For best performance, set the default cublas compute type to
  // CUBLAS_COMPUTE_16F for half precision and to
  // CUBLAS_COMPUTE_32F_FAST_16F for full precision
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
  if (m->output_type[0] == DT_FLOAT) {
    compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
  }
#endif
  checkCUDA(cublasGemmEx(m->handle.blas,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         out_dim,
                         batch_size,
                         in_dim,
                         &alpha,
                         m->offload ? m->weight_ptr : weight_ptr,
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
    // fuse bias and relu
    if (m->activation == AC_MODE_RELU) {
      int parallelism = out_dim * batch_size;
      AddBiasWithReLU<<<GET_BLOCKS(parallelism), CUDA_NUM_THREADS, 0, stream>>>(
          static_cast<DT *>(output_ptr),
          static_cast<DT const *>(bias_ptr),
          out_dim,
          batch_size);
      return;
    }
    checkCUDA(cublasGemmEx(m->handle.blas,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
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
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  if (use_activation(m->activation)) {
    checkCUDNN(cudnnActivationForward(m->handle.dnn,
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
    gelu_forward_kernel<<<GET_BLOCKS(elements), CUDA_NUM_THREADS>>>(
        elements, B, C, (float *)output_ptr);
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
                     ffStream_t stream) {
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  DT alpha = 1.0f;
  float sgeam_alpha = 1.0f;
  cudaDataType_t input_type = ff_to_cuda_datatype(m->input_type[0]);
  cudaDataType_t weight_type = ff_to_cuda_datatype(m->weight_type[0]);
  cudaDataType_t output_type = ff_to_cuda_datatype(m->output_type[0]);
#if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
  cudaDataType_t compute_type = cublas_data_type;
#else
  // For best performance, set the default cublas compute type to
  // CUBLAS_COMPUTE_16F for half precision and to
  // CUBLAS_COMPUTE_32F_FAST_16F for full precision
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
  if (m->output_type[0] == DT_FLOAT) {
    compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
  }
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
  checkCUDA(cublasGemmEx(m->handle.blas,
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
  if (m->kernel_reg_type == REG_MODE_NONE) {
    // do nothing
  } else if (m->kernel_reg_type == REG_MODE_L2) {
    checkCUDA(cublasSgeam(m->handle.blas,
                          CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          in_dim,
                          out_dim,
                          &sgeam_alpha,
                          (float *)kernel_grad_ptr,
                          in_dim,
                          &(m->kernel_reg_lambda),
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
    checkCUDA(cublasGemmEx(m->handle.blas,
                           CUBLAS_OP_N,
                           CUBLAS_OP_T,
                           1,
                           out_dim,
                           batch_size,
                           &alpha,
                           static_cast<DT *>(m->one_ptr),
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
    checkCUDA(cublasGemmEx(m->handle.blas,
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

template <typename DT>
__global__ void build_one_ptr(DT *one_ptr, int batch_size) {
  CUDA_KERNEL_LOOP(i, batch_size) {
    one_ptr[i] = static_cast<DT>(1.0f);
  }
}

} // namespace Internal
} // namespace Linear
} // namespace Kernels
} // namespace FlexFlow

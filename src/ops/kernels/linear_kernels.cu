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
#include "flexflow/initializer.h"
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

#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
std::mutex LinearMeta::profile_lock;
std::unordered_map<cublasAlgoConfig_t, int, cublasAlgoConfig_hasher>
    LinearMeta::algo_map;

void LinearMeta::findBestAlgoID(int m, int n, int k) {
  // return; // NO_PROF

  std::lock_guard<std::mutex> lock(profile_lock);

  cudaDataType_t cublas_data_type = ff_to_cuda_datatype(output_type[0]);
#if CUDA_VERSION >= 11000
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
  cudaDataType_t compute_type = cublas_data_type;
#endif

  cublasAlgoConfig_t mark{1, m, n, k, (int)compute_type};
  auto iter = algo_map.find(mark);
  if (iter != algo_map.end()) {
    return;
  }

  // don't find, profile required
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDA(cublasSetStream(handle.blas, stream));
  checkCUDNN(cudnnSetStream(handle.dnn, stream));
  switch (output_type[0]) {
    case DT_FLOAT:
      cudaRandomUniform(static_cast<float *>(handle.workSpace),
                        handle.workSpaceSize / data_type_size(DT_FLOAT));
      break;
    case DT_HALF:
      cudaRandomUniform(static_cast<half *>(handle.workSpace),
                        handle.workSpaceSize / data_type_size(DT_HALF));
      break;
    default:
      assert(false);
      break;
  }

  float alpha = 1.0f, beta = 0.0f;

  int startAlgo, endAlgo;
  int const ites = 100;
  struct timeval start, end;
  // TODO: only support 16F
  if (compute_type == CUBLAS_COMPUTE_16F) {
    startAlgo = (int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    endAlgo = (int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
  } else if (compute_type == CUDA_R_32F) {
    startAlgo = (int)CUBLAS_GEMM_DEFAULT;
    endAlgo = (int)CUBLAS_GEMM_ALGO23;
  } else {
    assert(false);
  }

  printf("***Cublas Gemm Testing Begin (Linear)***\n");
  printf("\n-----------------------------\n");
  printf("GEMM test: [M: %d, K: %d, N: %d] for linear\n", m, k, n);
  // todo
  void *d_A = handle.workSpace;
  void *d_B = d_A + m * k * data_type_size(output_type[0]);
  void *d_C = d_B + k * n * data_type_size(output_type[0]);

  float exec_time = 99999.0f;
  int fast_algo = 0;

  for (int algo = startAlgo; algo <= endAlgo; algo++) {
    cublasStatus_t status;
    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);
    for (int ite = 0; ite < ites; ++ite) {
      // checkCUDA(cublasGemmEx(m->handle.blas,
      //                    CUBLAS_OP_T,
      //                    CUBLAS_OP_N,
      //                    out_dim,
      //                    batch_size,
      //                    in_dim,
      //                    &alpha,
      //                    m->offload ? m->weight_ptr : weight_ptr,
      //                    weight_type,
      //                    in_dim,
      //                    input_ptr,
      //                    input_type,
      //                    in_dim,
      //                    &beta,
      //                    output_ptr,
      //                    output_type,
      //                    out_dim,
      //                    compute_type,
      //                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      status = cublasGemmEx(handle.blas,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            m, // out_dim
                            n, // batch_size
                            k, // in_dim
                            &alpha,
                            d_A,
                            cublas_data_type,
                            k,
                            d_B,
                            cublas_data_type,
                            k,
                            &beta,
                            d_C,
                            cublas_data_type,
                            m,
                            compute_type,
                            static_cast<cublasGemmAlgo_t>(algo));

      if (status != CUBLAS_STATUS_SUCCESS) {
        break;
      }
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    if (status == CUBLAS_STATUS_SUCCESS) {
      printf("algo_%d costs %.3fms \n", algo, diffTime(start, end) / ites);
      if (diffTime(start, end) / ites < exec_time) {
        exec_time = diffTime(start, end) / ites;
        fast_algo = algo;
      }
    }
  }

  printf("fast_algo %d costs %.3f ms\n", fast_algo, exec_time);
  algo_map.insert(std::pair<const cublasAlgoConfig_t, int>(mark, fast_algo));
}
#endif
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
#if CUDA_VERSION >= 11000
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
  cudaDataType_t compute_type = input_type;
#endif
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cublasAlgoConfig_t mark{1, out_dim, batch_size, in_dim, (int)compute_type};
  auto iter = m->algo_map.find(mark);
  cublasGemmAlgo_t algo_best =
      iter == m->algo_map.end() ? CUBLAS_GEMM_DEFAULT_TENSOR_OP
                                : static_cast<cublasGemmAlgo_t>(iter->second);
#else
  cublasGemmAlgo_t algo_best = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
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
                         algo_best));
  // use_bias = True
  if (bias_ptr != NULL) {
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
    mark.k = 1;
    auto iter = m->algo_map.find(mark);
    algo_best = iter == m->algo_map.end()
                    ? CUBLAS_GEMM_DEFAULT_TENSOR_OP
                    : static_cast<cublasGemmAlgo_t>(iter->second);
#else
    algo_best = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
#endif
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
                           algo_best));
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
#if CUDA_VERSION >= 11000
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
  cudaDataType_t compute_type = CUDA_R_32F;
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

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
#include <random>
#include <vector>

namespace FlexFlow {

LoraLinearMeta::LoraLinearMeta(FFHandler handler, LoraLinear const *li)
    : OpMeta(handler, li) {
  allocated_peft_buffer_size1 = 0;
  allocated_peft_buffer_size2 = 0;
}

LoraLinearMeta::~LoraLinearMeta(void) {}

namespace Kernels {
namespace LoraLinear {

void init_kernel_wrapper(LoraLinearMeta *m, int seed) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  if (m->input_type[0] == DT_FLOAT) {
    Internal::init_kernel<float>(m, seed, stream);
  } else if (m->input_type[0] == DT_HALF) {
    Internal::init_kernel<half>(m, seed, stream);
  } else {
    assert(false && "Unsupported data type");
  }
}

void inference_kernel_wrapper(LoraLinearMeta *m,
                              BatchConfig const *bc,
                              GenericTensorAccessorR const &input,
                              GenericTensorAccessorW const &output) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  int in_dim = input.domain.hi()[0] - input.domain.lo()[0] + 1;
  int out_dim = output.domain.hi()[0] - output.domain.lo()[0] + 1;

  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  if (m->input_type[0] == DT_FLOAT) {
    Internal::inference_kernel<float>(m,
                                      bc,
                                      input.get_float_ptr(),
                                      output.get_float_ptr(),
                                      in_dim,
                                      out_dim,
                                      stream);
  } else if (m->input_type[0] == DT_HALF) {
    Internal::inference_kernel<half>(m,
                                     bc,
                                     input.get_half_ptr(),
                                     output.get_half_ptr(),
                                     in_dim,
                                     out_dim,
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

void peft_bwd_kernel_wrapper(Context ctx,
                             Runtime *runtime,
                             LoraLinearMeta *m,
                             BatchConfig const *bc,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &output_grad) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  int in_dim = input_grad.domain.hi()[0] - input_grad.domain.lo()[0] + 1;
  int out_dim = output_grad.domain.hi()[0] - output_grad.domain.lo()[0] + 1;
  if (m->input_type[0] == DT_FLOAT) {
    Internal::peft_bwd_kernel<float>(ctx,
                                     runtime,
                                     m,
                                     bc,
                                     input_grad.get_float_ptr(),
                                     output_grad.get_float_ptr(),
                                     in_dim,
                                     out_dim,
                                     stream);
  } else if (m->input_type[0] == DT_HALF) {
    Internal::peft_bwd_kernel<half>(ctx,
                                    runtime,
                                    m,
                                    bc,
                                    input_grad.get_half_ptr(),
                                    output_grad.get_half_ptr(),
                                    in_dim,
                                    out_dim,
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
void init_kernel(LoraLinearMeta *m, int seed, cudaStream_t stream) {
  // Initialize generator
  std::mt19937 gen(seed);

  // Get handle to weights by iterating over m->model_state to get each
  // LoraLinearWeight object
  for (auto &model_state : m->model_state) {
    LoraLinearWeight weight = model_state.second.weights;
    int w0_num_elements = weight.rank * weight.in_dim;
    int w1_num_elements = weight.rank * weight.out_dim;

    // LoRA_A weight: [in_dim, rank]
    float stdv_lora_a = 1.0f / sqrt(weight.in_dim);
    std::uniform_real_distribution<float> dis_lora_a(-stdv_lora_a, stdv_lora_a);
    std::vector<DT> lora_a_random_init(w0_num_elements);
    for (auto &num : lora_a_random_init) {
      float num_float = dis_lora_a(gen);
      if (std::is_same<DT, half>::value) {
        num = __float2half(num_float);
      } else {
        num = num_float;
      }
    }
    checkCUDA(cudaMemcpyAsync(static_cast<DT *>(weight.w0_ptr),
                              lora_a_random_init.data(),
                              w0_num_elements * sizeof(DT),
                              cudaMemcpyHostToDevice,
                              stream));

    // LoRA_B weight: [rank, out_dim]
    float stdv_lora_b = 1.0f / sqrt(weight.rank);
    std::uniform_real_distribution<float> dis_lora_b(-stdv_lora_b, stdv_lora_b);
    std::vector<float> lora_b_random_init(w1_num_elements);
    for (auto &num : lora_b_random_init) {
      float num_float = dis_lora_b(gen);
      if (std::is_same<DT, half>::value) {
        num = __float2half(num_float);
      } else {
        num = num_float;
      }
    }
    checkCUDA(cudaMemcpyAsync(static_cast<DT *>(weight.w1_ptr),
                              lora_b_random_init.data(),
                              w1_num_elements * sizeof(DT),
                              cudaMemcpyHostToDevice,
                              stream));
  }
}

template <typename DT>
void inference_kernel(LoraLinearMeta *m,
                      BatchConfig const *bc,
                      DT const *input_ptr,
                      DT *output_ptr,
                      int in_dim,
                      int out_dim,
                      ffStream_t stream) {
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  DT alpha = 1.0f, beta = 0.0f;
  cudaDataType_t input_type = ff_to_cuda_datatype(m->input_type[0]);
  cudaDataType_t output_type = ff_to_cuda_datatype(m->input_type[1]);
  cudaDataType_t lr_actv_type = output_type;
  assert(input_type == output_type);
  cudaDataType_t weight_type = output_type;
  cudaDataType_t compute_type = output_type;
  // #if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
  //   cudaDataType_t compute_type = output_type;
  // #else
  //   // For best performance, set the default cublas compute type to
  //   // CUBLAS_COMPUTE_16F for half precision and to
  //   // CUBLAS_COMPUTE_32F_FAST_16F for full precision
  //   cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
  //   if (m->input_type[0] == DT_FLOAT) {
  //     compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
  //   }
  // #endif
  int num_peft_requests = 0;
  for (int i = 0; i < bc->max_requests_per_batch(); i++) {
    if (bc->request_completed[i]) {
      continue;
    }
    if (bc->requestsInfo[i].peft_model_id == PEFTModelID::NO_ID) {
      continue;
    }
    if (bc->requestsInfo[i].peft_bwd) {
      num_peft_requests++;
    }
  }
  // Assert that we have at most one request that requires peft_bwd
  assert(num_peft_requests <= 1);
  for (int i = 0; i < bc->max_requests_per_batch(); i++) {
    if (bc->request_completed[i]) {
      continue;
    }
    // Skip non-PEFT requests
    if (bc->requestsInfo[i].peft_model_id == PEFTModelID::NO_ID) {
      continue;
    }
    int num_peft_tokens = bc->requestsInfo[i].num_tokens_in_batch;
    int max_peft_tokens = bc->requestsInfo[i].max_length;
    int first_token_offset = bc->requestsInfo[i].first_token_offset_in_batch;
    assert(m->model_state.find(bc->requestsInfo[i].peft_model_id) !=
           m->model_state.end());
    LoraLinearWeight weight =
        m->model_state[bc->requestsInfo[i].peft_model_id].weights;
    int rank = weight.rank;
    void *intermediate_result_ptr = nullptr;
    if (bc->requestsInfo[i].peft_bwd) {
      size_t activation_size_needed1 =
          data_type_size(m->input_type[0]) * max_peft_tokens * in_dim;
      size_t activation_size_needed2 =
          data_type_size(m->input_type[1]) * max_peft_tokens * rank;
      MemoryAllocator *allocator = m->handle.peft_activation_allocator;
      if (activation_size_needed1 > m->allocated_peft_buffer_size1) {
        m->input_activation =
            allocator->allocate_instance_untyped(activation_size_needed1);
        m->allocated_peft_buffer_size1 = activation_size_needed1;
      }
      if (activation_size_needed2 > m->allocated_peft_buffer_size2) {
        m->low_rank_activation =
            allocator->allocate_instance_untyped(activation_size_needed2);
        m->allocated_peft_buffer_size2 = activation_size_needed2;
      }
      // copy input activation
      checkCUDA(cudaMemcpyAsync(m->input_activation,
                                input_ptr + first_token_offset * in_dim,
                                data_type_size(m->input_type[0]) *
                                    num_peft_tokens * in_dim,
                                cudaMemcpyDeviceToDevice,
                                stream));
      intermediate_result_ptr = m->low_rank_activation;
    } else {
      // use workspace to save intermediate result
      assert(m->handle.workSpaceSize >=
             data_type_size(m->input_type[1]) * num_peft_tokens * rank);
      intermediate_result_ptr = m->handle.workSpace;
    }
    // buffer = weight_first * input
    // [rank, num_peft_tokens] = [in_dim, rank].T * [in_dim, num_peft_tokens]
    checkCUDA(cublasGemmEx(m->handle.blas,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           rank,
                           num_peft_tokens,
                           in_dim,
                           &alpha,
                           weight.w0_ptr,
                           weight_type,
                           in_dim,
                           input_ptr + first_token_offset * in_dim,
                           input_type,
                           in_dim,
                           &beta,
                           intermediate_result_ptr,
                           lr_actv_type,
                           rank,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    // output = weight_second * buffer
    // [out_dim, num_peft_tokens] = [rank, out_dim].T * [rank, num_peft_tokens]
    // Note that we use alpha in both places since we do
    // an in-place update for LoraLinear
    float lora_alpha =
        m->model_state[bc->requestsInfo[i].peft_model_id].lora_alpha;
    DT scaling_constant = (DT)(lora_alpha / rank);
    checkCUDA(cublasGemmEx(m->handle.blas,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           out_dim,
                           num_peft_tokens,
                           rank,
                           &scaling_constant,
                           weight.w1_ptr,
                           weight_type,
                           rank,
                           intermediate_result_ptr,
                           lr_actv_type,
                           rank,
                           &alpha,
                           output_ptr + first_token_offset * out_dim,
                           output_type,
                           out_dim,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
}

template <typename DT>
__global__ void sgd_update(size_t count,
                           float lr,
                           float weight_decay,
                           float momentum,
                           bool nesterov,
                           DT const *WGrad,
                           DT *V,
                           DT *W) {
  // Refernce https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
  CUDA_KERNEL_LOOP(i, count) {
    DT gt = WGrad[i] + (DT)weight_decay * W[i];
    if (momentum > 0.0f) {
      V[i] = V[i] * (DT)momentum + gt;
      if (nesterov) {
        gt = gt + (DT)momentum * V[i];
      } else {
        gt = V[i];
      }
    }
    W[i] -= (DT)lr * gt;
  }
}

template <typename DT>
void peft_bwd_kernel(Context ctx,
                      Runtime *runtime,
                      LoraLinearMeta *m,
                     BatchConfig const *bc,
                     DT *input_grad_ptr,
                     DT const *output_grad_ptr,
                     int in_dim,
                     int out_dim,
                     ffStream_t stream) {
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  cudaDataType_t input_type = ff_to_cuda_datatype(m->input_type[0]);
  cudaDataType_t output_type = ff_to_cuda_datatype(m->output_type[0]);
  assert(input_type == output_type);
  cudaDataType_t weight_type = output_type;
  cudaDataType_t lr_actv_type = output_type;
  cudaDataType_t compute_type = output_type;
  // #if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
  //   cudaDataType_t compute_type = output_type;
  // #else
  //   // For best performance, set the default cublas compute type to
  //   // CUBLAS_COMPUTE_16F for half precision and to
  //   // CUBLAS_COMPUTE_32F_FAST_16F for full precision
  //   cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
  //   if (m->output_type[0] == DT_FLOAT) {
  //     compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
  //   }
  // #endif
  for (int i = 0; i < bc->max_requests_per_batch(); i++) {
    if (bc->request_completed[i]) {
      continue;
    }
    // Skip non-PEFT requests
    if (bc->requestsInfo[i].peft_model_id == PEFTModelID::NO_ID) {
      continue;
    }
    // Skip PEFT forward-only requests
    if (!bc->requestsInfo[i].peft_bwd) {
      continue;
    }
    int num_peft_tokens = bc->requestsInfo[i].num_tokens_in_batch;
    // int first_token_offset = bc->requestsInfo[i].first_token_offset_in_batch;
    assert(m->model_state.find(bc->requestsInfo[i].peft_model_id) !=
           m->model_state.end());
    LoraLinearWeight weight =
        m->model_state[bc->requestsInfo[i].peft_model_id].weights;
    int rank = weight.rank;
    float lora_alpha =
        m->model_state[bc->requestsInfo[i].peft_model_id].lora_alpha;
    DT scaling_constant = (DT)(lora_alpha / rank);

    // Compute LORA_B weight's gradient
    if (bc->requestsInfo[i].optimizer_tasks.compute_gradients) {
      DT alpha = 1.0f;
      DT beta = (bc->requestsInfo[i].optimizer_tasks.reset_gradients_to_zero)
                    ? 0.0f
                    : 1.0f;
      checkCUDA(cublasGemmEx(m->handle.blas,
                             CUBLAS_OP_N,
                             CUBLAS_OP_T,
                             rank,
                             out_dim,
                             num_peft_tokens,
                             &scaling_constant,
                             m->low_rank_activation,
                             lr_actv_type,
                             rank,
                             output_grad_ptr,
                             output_type,
                             out_dim,
                             &beta,
                             weight.w1_grad_ptr,
                             weight_type,
                             rank,
                             compute_type,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Compute LORA_B input's (and LORA_A output's) gradient inplace in
    // low_rank_activation
    {
      DT alpha = 1.0f, beta = 0.0f;
      checkCUDA(cublasGemmEx(m->handle.blas,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             rank,
                             num_peft_tokens,
                             out_dim,
                             &scaling_constant,
                             weight.w1_ptr,
                             weight_type,
                             rank,
                             output_grad_ptr,
                             output_type,
                             out_dim,
                             &beta,
                             m->low_rank_activation,
                             lr_actv_type,
                             rank,
                             compute_type,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Compute LORA_A weight's gradient
    if (bc->requestsInfo[i].optimizer_tasks.compute_gradients) {
      DT alpha = 1.0f;
      DT beta = (bc->requestsInfo[i].optimizer_tasks.reset_gradients_to_zero)
                    ? 0.0f
                    : 1.0f;
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
                             &beta,
                             weight.w0_grad_ptr,
                             weight_type,
                             in_dim,
                             compute_type,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    // Compute input gradient
    // NOTE: we use beta=1 for input_grad to accumulate gradients when needed
    if (input_grad_ptr != nullptr) {
      DT alpha = 1.0f;
      DT beta = m->reset_input_grads[0] ? 0.0f : 1.0f;
      checkCUDA(cublasGemmEx(m->handle.blas,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             in_dim,
                             num_peft_tokens,
                             rank,
                             &alpha,
                             weight.w0_ptr,
                             weight_type,
                             in_dim,
                             m->low_rank_activation,
                             lr_actv_type,
                             rank,
                             &beta,
                             input_grad_ptr,
                             input_type,
                             in_dim,
                             compute_type,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    if (bc->requestsInfo[i].optimizer_tasks.update_weights) {
      LoraOptimizerConfig const *optimizer_config =
          m->model_state[bc->requestsInfo[i].peft_model_id].optimizer_config;
      assert(optimizer_config != nullptr);
      assert(typeid(*optimizer_config) != typeid(LoraOptimizerConfig));
      int w0_num_elements = rank * in_dim;
      int w1_num_elements = rank * out_dim;

      // Get optimizer config
      if (typeid(*optimizer_config) == typeid(LoraSGDOptimizerConfig)) {
        LoraSGDOptimizerConfig const *sgd_config =
            (LoraSGDOptimizerConfig const *)optimizer_config;
        // LoRA_A weight is split in tensor parallelism, so no need to apply
        // all-reduce
        sgd_update<<<GET_BLOCKS(w0_num_elements),
                     CUDA_NUM_THREADS,
                     0,
                     stream>>>(w0_num_elements,
                               sgd_config->lr,
                               sgd_config->weight_decay,
                               sgd_config->momentum,
                               sgd_config->nesterov,
                               static_cast<DT const *>(weight.w0_grad_ptr),
                               static_cast<DT *>(weight.w0_v_values_ptr),
                               static_cast<DT *>(weight.w0_ptr));
        // LoRA_B weight is replicated w tensor parallelism, so we need to sync
        // and sum first
#ifdef FF_USE_NCCL
        ncclDataType_t nccl_data_type = ff_to_nccl_datatype(m->output_type[0]);
        runtime->concurrent_task_barrier(ctx);
        checkNCCL(ncclAllReduce(static_cast<DT const *>(weight.w1_grad_ptr),
                                static_cast<DT *>(weight.w1_grad_ptr),
                                w1_num_elements,
                                nccl_data_type,
                                ncclSum,
                                m->handle.ncclComm,
                                stream));
        runtime->concurrent_task_barrier(ctx);
#else
        assert(false && "Must enable FF_USE_NCCL to use AllReduce operators");
#endif
        sgd_update<<<GET_BLOCKS(w1_num_elements),
                     CUDA_NUM_THREADS,
                     0,
                     stream>>>(w1_num_elements,
                               sgd_config->lr,
                               sgd_config->weight_decay,
                               sgd_config->momentum,
                               sgd_config->nesterov,
                               static_cast<DT const *>(weight.w1_grad_ptr),
                               static_cast<DT *>(weight.w1_v_values_ptr),
                               static_cast<DT *>(weight.w1_ptr));
      } else if (typeid(*optimizer_config) == typeid(LoraAdamOptimizerConfig)) {
        assert(false && "Adam optimizer type not implemented yet");
      } else {
        assert(false && "Unsupported optimizer type");
      }
    }
  }
}

} // namespace Internal
} // namespace LoraLinear
} // namespace Kernels
} // namespace FlexFlow

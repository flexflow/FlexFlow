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
#include "flexflow/ops/kernels/softmax_kernels.h"
#include "flexflow/utils/cuda_helper.h"
#include "flexflow/utils/hash_utils.h"

namespace FlexFlow {
// declare Legion names
using Legion::Domain;

SoftmaxMeta::SoftmaxMeta(FFHandler handler,
                         Softmax const *softmax,
                         MemoryAllocator &gpu_mem_allocator,
                         Domain const &input_domain)
    : OpMeta(handler, softmax) {
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain4SoftMax(
      inputTensor, input_domain, softmax->data_type));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain4SoftMax(
      outputTensor, input_domain, softmax->data_type));
  dim = softmax->dim;
  profiling = softmax->profiling;
  inference_debugging = softmax->inference_debugging;
  std::strcpy(op_name, softmax->name);

  int max_tokens_per_batch = BatchConfig::max_tokens_per_batch();

  // todo change this
  int vocab_size = 32000;
  size_t size_of_dt = data_type_size(softmax->data_type);
  size_t totalSize = max_tokens_per_batch * size_of_dt * vocab_size;
  gpu_mem_allocator.create_legion_instance(reserveInst, totalSize);
  lm_head_cache = gpu_mem_allocator.allocate_instance_untyped(totalSize);
}

SoftmaxMeta::~SoftmaxMeta(void) {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
}

namespace Kernels {
namespace Softmax {

void forward_kernel_wrapper(SoftmaxMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  if (m->output_type[0] == DT_FLOAT) {
    Internal::forward_kernel(
        m, input.get_float_ptr(), output.get_float_ptr(), stream);
  } else if (m->output_type[0] == DT_HALF) {
    Internal::forward_kernel(
        m, input.get_half_ptr(), output.get_half_ptr(), stream);
  } else {
    assert(false && "Unsupported data type");
  }
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    // print_tensor<float>(acc_input.ptr, acc_input.rect.volume(),
    // "[Softmax:forward:input]"); print_tensor<float>(acc_output.ptr,
    // acc_output.rect.volume(), "[Softmax:forward:output]");
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    log_measure.debug(
        "%s [Softmax] forward time = %.2fms\n", m->op_name, elapsed);
  }
}

void backward_kernel_wrapper(SoftmaxMeta const *m,
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
  assert(input_grad.domain == output_grad.domain);
  if (m->output_type[0] == DT_FLOAT) {
    Internal::backward_kernel(m,
                              input_grad.get_float_ptr(),
                              output_grad.get_float_ptr(),
                              output_grad.domain.get_volume(),
                              stream);
  } else if (m->output_type[0] == DT_HALF) {
    Internal::backward_kernel(m,
                              input_grad.get_half_ptr(),
                              output_grad.get_half_ptr(),
                              output_grad.domain.get_volume(),
                              stream);
  } else {
    assert(false && "Unsupported data type");
  }
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    // print_tensor<float>(acc_output_grad.ptr, acc_output_grad.rect.volume(),
    // "[Softmax:backward:output_grad]");
    // print_tensor<float>(acc_input_grad.ptr, acc_input_grad.rect.volume(),
    // "[Softmax:backward:input_grad]");
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    log_measure.debug("Softmax backward time = %.2fms\n", elapsed);
  }
}

void inference_kernel_wrapper(SoftmaxMeta const *m,
                              BatchConfig const *bc,
                              GenericTensorAccessorR const &input,
                              GenericTensorAccessorW const &output,
                              GenericTensorAccessorW const &output_grad) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  int num_classes = output.domain.hi()[0] - output.domain.lo()[0] + 1;

  if (m->output_type[0] == DT_FLOAT) {
    Internal::inference_kernel(m,
                               bc,
                               input.get_float_ptr(),
                               output.get_float_ptr(),
                               num_classes,
                               stream);
    checkCUDA(cudaMemcpyAsync(output_grad.get_float_ptr(),
                              output.get_float_ptr(),
                              output.domain.get_volume() * sizeof(float),
                              cudaMemcpyDeviceToDevice,
                              stream));
  } else if (m->output_type[0] == DT_HALF) {
    Internal::inference_kernel(m,
                               bc,
                               input.get_half_ptr(),
                               output.get_half_ptr(),
                               num_classes,
                               stream);
    checkCUDA(cudaMemcpyAsync(output_grad.get_half_ptr(),
                              output.get_half_ptr(),
                              output.domain.get_volume() * sizeof(half),
                              cudaMemcpyDeviceToDevice,
                              stream));
  } else {
    assert(false && "Unsupported data type");
  }
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    // print_tensor<float>(acc_input.ptr, acc_input.rect.volume(),
    // "[Softmax:forward:input]"); print_tensor<float>(acc_output.ptr,
    // acc_output.rect.volume(), "[Softmax:forward:output]");
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    log_measure.debug(
        "%s [Softmax] inference time = %.2fms\n", m->op_name, elapsed);
  }
}

void peft_bwd_kernel_wrapper(SoftmaxMeta const *m,
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

  int num_classes = output_grad.domain.hi()[0] - output_grad.domain.lo()[0] + 1;
  if (m->output_type[0] == DT_FLOAT) {
    Internal::peft_bwd_kernel(m,
                              bc,
                              input_grad.get_float_ptr(),
                              output_grad.get_float_ptr(),
                              num_classes,
                              stream);
  } else if (m->output_type[0] == DT_HALF) {
    Internal::peft_bwd_kernel(m,
                              bc,
                              input_grad.get_half_ptr(),
                              output_grad.get_half_ptr(),
                              num_classes,
                              stream);
  } else {
    assert(false && "Unsupported data type");
  }
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    // print_tensor<float>(acc_input.ptr, acc_input.rect.volume(),
    // "[Softmax:forward:input]"); print_tensor<float>(acc_output.ptr,
    // acc_output.rect.volume(), "[Softmax:forward:output]");
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    log_measure.debug(
        "%s [Softmax] inference time = %.2fms\n", m->op_name, elapsed);
  }
}

namespace Internal {
template <typename DT>
void forward_kernel(SoftmaxMeta const *m,
                    DT const *input_ptr,
                    DT *output_ptr,
                    cudaStream_t stream) {
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnSoftmaxForward(m->handle.dnn,
                                 CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_CHANNEL,
                                 &alpha,
                                 m->inputTensor,
                                 input_ptr,
                                 &beta,
                                 m->outputTensor,
                                 output_ptr));
}

template <typename DT>
void backward_kernel(SoftmaxMeta const *m,
                     DT *input_grad_ptr,
                     DT const *output_grad_ptr,
                     size_t num_elements,
                     cudaStream_t stream) {
  checkCUDA(cudaMemcpyAsync(input_grad_ptr,
                            output_grad_ptr,
                            num_elements * sizeof(DT),
                            cudaMemcpyDeviceToDevice,
                            stream));
}

template <typename DT>
void inference_kernel(SoftmaxMeta const *m,
                      BatchConfig const *bc,
                      DT const *input_ptr,
                      DT *output_ptr,
                      int num_classes,
                      cudaStream_t stream) {
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  cudnnDataType_t cudnn_data_type = ff_to_cudnn_datatype(m->output_type[0]);
  checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        cudnn_data_type,
                                        bc->num_active_tokens(),
                                        num_classes,
                                        1,
                                        1));
  checkCUDNN(cudnnSoftmaxForward(m->handle.dnn,
                                 CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_CHANNEL,
                                 &alpha,
                                 m->outputTensor,
                                 input_ptr,
                                 &beta,
                                 m->outputTensor,
                                 output_ptr));

  // store partial lm_head output
  std::cout << "sftmax forwwwwd: " << bc->num_active_tokens() << ", "
            << bc->num_active_peft_fwd_tokens_() << ", "
            << bc->num_active_peft_tokens() << "\n";
  for (int i = 0; i < bc->max_requests_per_batch(); i++) {
    if (bc->request_completed[i]) {
      continue;
    }
    if (bc->requestsInfo[i].peft_model_id == PEFTModelID::NO_ID ||
        bc->requestsInfo[i].peft_bwd) {
      continue;
    }

    int num_tokens = bc->requestsInfo[i].num_tokens_in_batch;
    int processed_tokens = bc->requestsInfo[i].peft_fwd_tokens - num_tokens;
    int first_token_offset_in_batch =
        bc->requestsInfo[i].first_token_offset_in_batch;

    std::cout << "first_token_offset_in_batch: " << first_token_offset_in_batch
              << "\n";
    std::cout << "processed fwd tokens: " << processed_tokens << "\n";
    std::cout << "num tokens: " << num_tokens << "\n";
    // std::cout << "total tokens: " << bc->num_active_infr_tokens() << "\n";
    if (processed_tokens == 0) {
      save_tensor<float>(
          (float *)output_ptr + first_token_offset_in_batch * num_classes,
          num_tokens * num_classes,
          "/home/xinhaoc/FlexFlow/inference/output_tensors/sftmax.txt");
    }

    std::cout << "store fwd results: " << processed_tokens
              << ", peft fwd tokens: " << bc->requestsInfo[i].peft_model_id
              << "\n";

    checkCUDA(cudaMemcpyAsync(
        static_cast<DT *>(m->lm_head_cache) + processed_tokens * num_classes,
        output_ptr + first_token_offset_in_batch * num_classes,
        sizeof(DT) * num_tokens * num_classes,
        cudaMemcpyDeviceToDevice,
        stream));

    print_tensor<float>((float*) m->lm_head_cache + processed_tokens * num_classes, 32, "sft fffff");
  }
}

template <typename DT>
__global__ void sparse_categorical_crossentropy_loss_peft_backward(
    DT *input_grad,
    DT const *output_grad,
    BatchConfig::TokenId const *token_ids,
    int num_tokens,
    int num_classes) {
  CUDA_KERNEL_LOOP(i, num_tokens * num_classes) {
    input_grad[i] = output_grad[i];
    if (i % num_classes == token_ids[i / num_classes]) {
      input_grad[i] -= 1.0f;
    }
  }
}

template <typename DT>
void peft_bwd_kernel(SoftmaxMeta const *m,
                     BatchConfig const *bc,
                     DT *input_grad_ptr,
                     DT const *output_grad_ptr,
                     int num_classes,
                     cudaStream_t stream) {
  // save_tensor<float>((float*)m->lm_head_cache, 32000 * 10,
  // "/home/xinhaoc/FlexFlow/inference/output_tensors/xinhao_part.txt");
  BatchConfig::TokenId token_ids[BatchConfig::MAX_NUM_TOKENS];
  int tokens_previous_requests = 0;
  for (int i = 0; i < bc->max_requests_per_batch(); i++) {
    if (bc->request_completed[i]) {
      continue;
    }
    // Skip non-PEFT requests
    if (!bc->requestsInfo[i].peft_bwd) {
      tokens_previous_requests += bc->requestsInfo[i].num_tokens_in_batch;
      continue;
    }
    int num_bwd_tokens = bc->requestsInfo[i].num_tokens_in_batch;
    int num_processed_tokens =
        bc->requestsInfo[i].peft_bwd_tokens - num_bwd_tokens;
    for (int j = 0; j < num_bwd_tokens; j++) {
      token_ids[j] = bc->labelsInfo[j].token_id;
      printf("token id i %d, %d\n", j, token_ids[j]);
    }

    DT scale_factor = 1.0 / (bc->requestsInfo[i].peft_total_tokens - 1);
    // ignore last token
    // checkCUDA(cudaMemsetAsync(
    //       input_grad_ptr + (tokens_previous_requests) *
    //                            num_classes,
    //       DT(3),
    //       num_classes * num_bwd_tokens * sizeof(DT),
    //       stream));
    // assign_kernel<<<GET_BLOCKS(num_classes * num_bwd_tokens),
    //                 CUDA_NUM_THREADS,
    //                 0,
    //                 stream>>>(input_grad_ptr +
    //                               (tokens_previous_requests)*num_classes,
    //                           num_classes * num_bwd_tokens,
    //                           DT(0.1f));
    // printf("sss: %d", num_classes * num_bwd_tokens * sizeof(DT));

    if (num_processed_tokens == 0) {
      // assert(num_bwd_tokens + num_processed_tokens ==
      //        bc->requestsInfo[i].peft_total_tokens);
      checkCUDA(cudaMemsetAsync(
          input_grad_ptr + (tokens_previous_requests +
                            bc->requestsInfo[i].num_tokens_in_batch - 1) *
                               num_classes,
          0,
          num_classes * sizeof(DT),
          stream));
    }

    checkCUDA(cudaMemcpyAsync(m->handle.workSpace,
                              token_ids,
                              sizeof(BatchConfig::TokenId) * num_bwd_tokens,
                              cudaMemcpyHostToDevice,
                              stream));
    int start_offset = bc->requestsInfo[i].peft_total_tokens -
                       bc->requestsInfo[i].peft_bwd_tokens;

    //  std::cout <<
    // "bc->num_active_tokens(): " << num_bwd_tokens << ", " << start_offset  <<
    // "\n"; save_tensor<float>((float*)m->lm_head_cache + start_offset *
    // num_classes, num_classes * num_bwd_tokens,
    // "/home/xinhaoc/FlexFlow/inference/output_tensors/bwinput.txt");

    std::cout << "start_offset: " << start_offset << "\n";

    print_tensor<float>(
        (float *)m->lm_head_cache + start_offset * num_classes, 32, "grad");

    sparse_categorical_crossentropy_loss_peft_backward<<<
        GET_BLOCKS(num_bwd_tokens * num_classes),
        CUDA_NUM_THREADS,
        0,
        stream>>>(
        input_grad_ptr + tokens_previous_requests * num_classes,
        static_cast<DT *>(m->lm_head_cache) + start_offset * num_classes,
        static_cast<BatchConfig::TokenId const *>(m->handle.workSpace),
        num_processed_tokens == 0 ? num_bwd_tokens - 1 : num_bwd_tokens,
        num_classes);
    // scale
    scale_kernel<<<GET_BLOCKS(num_bwd_tokens * num_classes),
                   CUDA_NUM_THREADS,
                   0,
                   stream>>>(input_grad_ptr +
                                 tokens_previous_requests * num_classes,
                             num_bwd_tokens * num_classes,
                             DT(0.0),
                             scale_factor);
    save_tensor<float>(
        (float *)input_grad_ptr + tokens_previous_requests * num_classes,
        32000 * 5,
        "/home/xinhaoc/FlexFlow/inference/output_tensors/bwinput.txt");
    print_tensor<float>((float *)input_grad_ptr +
                            tokens_previous_requests * num_classes,
                        32,
                        "sftmax bw");
    tokens_previous_requests += num_bwd_tokens;

    // for(int k = 0; k < 5; k++){
    //     printf("token id i %d, %d\n", token_ids[i]);
    // }
  }
  assert(tokens_previous_requests == bc->num_active_tokens());
}

} // namespace Internal
} // namespace Softmax
} // namespace Kernels
} // namespace FlexFlow

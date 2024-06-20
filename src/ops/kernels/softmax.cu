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

#include "flexflow/ops/kernels/softmax_kernels.h"
#include "flexflow/utils/cuda_helper.h"
#include "flexflow/utils/hash_utils.h"

namespace FlexFlow {
// declare Legion names
using Legion::Domain;

SoftmaxMeta::SoftmaxMeta(FFHandler handler,
                         Softmax const *softmax,
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
                              bool is_last_op,
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
    if (is_last_op) {
      checkCUDA(cudaMemcpyAsync(output_grad.get_float_ptr(),
                                output.get_float_ptr(),
                                output.domain.get_volume() * sizeof(float),
                                cudaMemcpyDeviceToDevice,
                                stream));
    }
  } else if (m->output_type[0] == DT_HALF) {
    Internal::inference_kernel(m,
                               bc,
                               input.get_half_ptr(),
                               output.get_half_ptr(),
                               num_classes,
                               stream);
    if (is_last_op) {
      checkCUDA(cudaMemcpyAsync(output_grad.get_half_ptr(),
                                output.get_half_ptr(),
                                output.domain.get_volume() * sizeof(half),
                                cudaMemcpyDeviceToDevice,
                                stream));
    }
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
}

template <typename DT>
__global__ void sparse_categorical_crossentropy_loss_peft_backward(
    DT *input_grad,
    DT const *output_grad,
    BatchConfig::TokenId const *token_ids,
    int num_tokens,
    int num_classes) {
  CUDA_KERNEL_LOOP(i, num_tokens * num_classes) {
    int class_idx = i % num_classes;
    int token_idx = i / num_classes;
    input_grad[i] = output_grad[i];
    if (class_idx == token_ids[token_idx]) {
      input_grad[i] = input_grad[i] - (DT)1.0f;
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
    int num_bwd_tokens = bc->requestsInfo[i].num_tokens_in_batch - 1;
    // shift labels by 1 position to the left (ignore first token label)
    for (int j = 0; j < num_bwd_tokens; j++) {
      token_ids[j] = bc->tokensInfo[j + tokens_previous_requests + 1].token_id;
    }

    DT scale_factor = 1.0 / (bc->requestsInfo[i].num_tokens_in_batch - 1);
    // ignore last token
    checkCUDA(cudaMemsetAsync(
        input_grad_ptr + (tokens_previous_requests +
                          bc->requestsInfo[i].num_tokens_in_batch - 1) *
                             num_classes,
        0,
        num_classes * sizeof(DT),
        stream));
    checkCUDA(cudaMemcpyAsync(m->handle.workSpace,
                              token_ids,
                              sizeof(BatchConfig::TokenId) * num_bwd_tokens,
                              cudaMemcpyHostToDevice,
                              stream));
    sparse_categorical_crossentropy_loss_peft_backward<<<
        GET_BLOCKS(num_bwd_tokens * num_classes),
        CUDA_NUM_THREADS,
        0,
        stream>>>(
        input_grad_ptr + tokens_previous_requests * num_classes,
        output_grad_ptr + tokens_previous_requests * num_classes,
        static_cast<BatchConfig::TokenId const *>(m->handle.workSpace),
        num_bwd_tokens,
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

    tokens_previous_requests += num_bwd_tokens + 1;
  }
  assert(tokens_previous_requests == bc->num_active_tokens());
}

} // namespace Internal
} // namespace Softmax
} // namespace Kernels
} // namespace FlexFlow

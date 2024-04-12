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
#include "flexflow/ops/sigmoid_silu_multi.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

SigmoidSiluMultiMeta::SigmoidSiluMultiMeta(FFHandler handle,
                                           SigmoidSiluMulti const *ssm,
                                           MemoryAllocator &gpu_mem_allocator)
    : OpMeta(handle, ssm) {
  profiling = ssm->profiling;
  inference_debugging = ssm->inference_debugging;
  size_t in_dim = ssm->data_dim;

  size_t totalSize = 2 * BatchConfig::max_sequence_length() * in_dim *
                     data_type_size(ssm->data_type);
  gpu_mem_allocator.create_legion_instance(reserveInst, totalSize);
  input_activation = gpu_mem_allocator.allocate_instance_untyped(totalSize);
}

SigmoidSiluMultiMeta::~SigmoidSiluMultiMeta(void) {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
}

template <typename T>
__global__ void SigmoidSiluMultiKernel(int num_elements,
                                       T const *input1_ptr,
                                       T const *input2_ptr,
                                       T *output_ptr) {
  CUDA_KERNEL_LOOP(i, num_elements) {
    float sigmoid_val = static_cast<float>(input1_ptr[i]);
    sigmoid_val = 1.0f / (1.0f + exp(-sigmoid_val));
    output_ptr[i] = input1_ptr[i] * T(sigmoid_val) * input2_ptr[i];
  }
}

template <typename T>
__global__ void SigmoidSiluMultiBackwardKernel(int num_elements,
                                               T const *output_grad_ptr,
                                               T const *input1_ptr,
                                               T const *input2_ptr,
                                               T *input1_grad_ptr,
                                               T *input2_grad_ptr,
                                               bool reset_input_grad1,
                                               bool reset_input_grad2) {
  CUDA_KERNEL_LOOP(i, num_elements) {
    float sigmoid_val = static_cast<float>(input1_ptr[i]);
    sigmoid_val = 1.0f / (1.0f + exp(-sigmoid_val));

    if (reset_input_grad2) {
      input2_grad_ptr[i] =
          output_grad_ptr[i] * (input1_ptr[i] * T(sigmoid_val));
    } else {
      input2_grad_ptr[i] +=
          output_grad_ptr[i] * (input1_ptr[i] * T(sigmoid_val));
    }
    T ss_grad_val = output_grad_ptr[i] * input2_ptr[i];
    if (reset_input_grad1) {
      input1_grad_ptr[i] = ss_grad_val * T(sigmoid_val);
    } else {
      input1_grad_ptr[i] += ss_grad_val * T(sigmoid_val);
    }
    T sig_grad = ss_grad_val * input1_ptr[i];

    float x1_grad_val = static_cast<float>(sig_grad);
    x1_grad_val = x1_grad_val * sigmoid_val * (1.0f - sigmoid_val);
    input1_grad_ptr[i] += T(x1_grad_val);
  }
}

/*static*/
void SigmoidSiluMulti::inference_kernel_wrapper(
    SigmoidSiluMultiMeta *m,
    BatchConfig const *bc,
    GenericTensorAccessorR const &input1,
    GenericTensorAccessorR const &input2,
    GenericTensorAccessorW const &output) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  int num_elements = input1.domain.get_volume();
  assert(input2.domain.get_volume() == num_elements);
  assert(output.domain.get_volume() == num_elements);

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  // save input activation if needed for PEFT
  if (bc->num_active_peft_fwd_tokens_() > 0) {
    // Check that we have at most one request that requires peft_bwd
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
    assert(num_peft_requests <= 1);

    int tokens_previous_requests = 0;
    for (int i = 0; i < bc->max_requests_per_batch(); i++) {
      if (bc->request_completed[i]) {
        continue;
      }
      // Skip non-PEFT requests
      if (bc->requestsInfo[i].peft_model_id == PEFTModelID::NO_ID) {
        // FIXME: use the new approach to computing token offset
        tokens_previous_requests += bc->requestsInfo[i].num_tokens_in_batch;
        continue;
      }
      int num_peft_tokens = bc->requestsInfo[i].num_tokens_in_batch;
      int in_dim = input1.domain.hi()[0] - input1.domain.lo()[0] + 1;
      int num_processed_tokens =
          bc->requestsInfo[i].peft_fwd_tokens - num_peft_tokens;

      if (bc->requestsInfo[i].peft_bwd) {
        // MemoryAllocator *allocator = m->handle.peft_activation_allocator;
        size_t input_tensor_size =
            data_type_size(m->input_type[0]) * num_peft_tokens * in_dim;
        // m->input_activation =
        //     allocator->allocate_instance_untyped(2 * input_tensor_size);
        // copy input activation
        if (m->input_type[0] == DT_FLOAT) {
          checkCUDA(cudaMemcpyAsync(
              m->input_activation + num_processed_tokens *in_dim *
                                        data_type_size(m->input_type[0]),
              input1.get_float_ptr() + tokens_previous_requests * in_dim,
              input_tensor_size,
              cudaMemcpyDeviceToDevice,
              stream));
          checkCUDA(cudaMemcpyAsync(
              m->input_activation +
                  (BatchConfig::max_sequence_length() + num_processed_tokens) *
                      in_dim * data_type_size(m->input_type[0]),
              input2.get_float_ptr() + tokens_previous_requests * in_dim,
              input_tensor_size,
              cudaMemcpyDeviceToDevice,
              stream));
        } else if (m->input_type[0] == DT_HALF) {
          checkCUDA(cudaMemcpyAsync(m->input_activation,
                                    input1.get_half_ptr() +
                                        tokens_previous_requests * in_dim,
                                    input_tensor_size,
                                    cudaMemcpyDeviceToDevice,
                                    stream));
          checkCUDA(cudaMemcpyAsync(
              m->input_activation + BatchConfig::max_sequence_length() *
                                        in_dim *
                                        data_type_size(m->input_type[0]),
              input2.get_half_ptr() + tokens_previous_requests * in_dim,
              input_tensor_size,
              cudaMemcpyDeviceToDevice,
              stream));
        } else {
          assert(false && "unsupport datatype in layernorm");
        }
      }
    }
  }

  if (m->input_type[0] == DT_FLOAT) {
    SigmoidSiluMultiKernel<<<GET_BLOCKS(num_elements),
                             min(CUDA_NUM_THREADS, num_elements),
                             0,
                             stream>>>(input1.domain.get_volume(),
                                       input1.get_float_ptr(),
                                       input2.get_float_ptr(),
                                       output.get_float_ptr());
  } else if (m->input_type[0] == DT_HALF) {
    SigmoidSiluMultiKernel<<<GET_BLOCKS(num_elements),
                             min(CUDA_NUM_THREADS, num_elements),
                             0,
                             stream>>>(input1.domain.get_volume(),
                                       input1.get_half_ptr(),
                                       input2.get_half_ptr(),
                                       output.get_half_ptr());
  } else {
    assert(false && "unsupport datatype in SigmoidSiluMulti");
  }

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[SigmoidSiluMulti] forward time (CF) = %.9fms\n", elapsed);
  }
}

/*static*/
void SigmoidSiluMulti::backward_kernel_wrapper(
    SigmoidSiluMultiMeta const *m,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input1,
    GenericTensorAccessorR const &input2,
    GenericTensorAccessorW const &input1_grad,
    GenericTensorAccessorW const &input2_grad) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  int num_elements = output_grad.domain.get_volume();
  assert(input1.domain.get_volume() == num_elements);
  assert(input2.domain.get_volume() == num_elements);
  assert(input1_grad.domain.get_volume() == num_elements);
  assert(input2_grad.domain.get_volume() == num_elements);

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  if (m->input_type[0] == DT_FLOAT) {
    SigmoidSiluMultiBackwardKernel<<<GET_BLOCKS(num_elements),
                                     min(CUDA_NUM_THREADS, num_elements),
                                     0,
                                     stream>>>(output_grad.domain.get_volume(),
                                               output_grad.get_float_ptr(),
                                               input1.get_float_ptr(),
                                               input2.get_float_ptr(),
                                               input1_grad.get_float_ptr(),
                                               input2_grad.get_float_ptr(),
                                               m->reset_input_grads[0],
                                               m->reset_input_grads[1]);
  } else if (m->input_type[0] == DT_HALF) {
    SigmoidSiluMultiBackwardKernel<<<GET_BLOCKS(num_elements),
                                     min(CUDA_NUM_THREADS, num_elements),
                                     0,
                                     stream>>>(output_grad.domain.get_volume(),
                                               output_grad.get_half_ptr(),
                                               input1.get_half_ptr(),
                                               input2.get_half_ptr(),
                                               input1_grad.get_half_ptr(),
                                               input2_grad.get_half_ptr(),
                                               m->reset_input_grads[0],
                                               m->reset_input_grads[1]);
  } else {
    assert(false && "unsupport datatype in SigmoidSiluMulti");
  }
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[SigmoidSiluMulti] backward time (CF) = %.9fms\n", elapsed);
  }
}

/*static*/
void SigmoidSiluMulti::peft_bwd_kernel_wrapper(
    SigmoidSiluMultiMeta const *m,
    BatchConfig const *bc,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorW const &input1_grad,
    GenericTensorAccessorW const &input2_grad) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  assert(input1_grad.domain.get_volume() == output_grad.domain.get_volume());
  assert(input2_grad.domain.get_volume() == input1_grad.domain.get_volume());

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  int num_peft_requests = 0;
  int num_peft_tokens = 0;
  for (int i = 0; i < bc->max_requests_per_batch(); i++) {
    if (bc->request_completed[i]) {
      continue;
    }
    if (bc->requestsInfo[i].peft_model_id == PEFTModelID::NO_ID ||
        (!bc->requestsInfo[i].peft_bwd)) {
      continue;
    }
    num_peft_requests++;
    num_peft_tokens = bc->requestsInfo[i].num_tokens_in_batch;
    int in_dim = output_grad.domain.hi()[0] - output_grad.domain.lo()[0] + 1;
    int num_elements = in_dim * num_peft_tokens;

    printf("silu: %d\n", num_peft_tokens);
    save_tensor<float>((float*)output_grad.get_float_ptr() + 1 * in_dim, in_dim * num_peft_tokens, "/home/xinhaoc/FlexFlow/inference/output/silu.txt");

    
    int num_total_tokens = bc->requestsInfo[i].peft_fwd_tokens;
    int token_start_offset =
        num_total_tokens - bc->requestsInfo[i].peft_bwd_tokens;

    if (m->input_type[0] == DT_FLOAT) {
      SigmoidSiluMultiBackwardKernel<<<GET_BLOCKS(num_elements),
                                       min(CUDA_NUM_THREADS, num_elements),
                                       0,
                                       stream>>>(
          num_elements,
          output_grad.get_float_ptr(),
          static_cast<float const *>(m->input_activation) +
              token_start_offset * in_dim,
          static_cast<float const *>(m->input_activation) +
              (BatchConfig::max_sequence_length() + token_start_offset) *
                  in_dim,
          input1_grad.get_float_ptr(),
          input2_grad.get_float_ptr(),
          m->reset_input_grads[0],
          m->reset_input_grads[1]);
    } else if (m->input_type[0] == DT_HALF) {
      SigmoidSiluMultiBackwardKernel<<<GET_BLOCKS(num_elements),
                                       min(CUDA_NUM_THREADS, num_elements),
                                       0,
                                       stream>>>(
          num_elements,
          output_grad.get_half_ptr(),
          static_cast<half const *>(m->input_activation) +
              token_start_offset * in_dim,
          static_cast<half const *>(m->input_activation) +
              (BatchConfig::max_sequence_length() + token_start_offset) *
                  in_dim,
          input1_grad.get_half_ptr(),
          input2_grad.get_half_ptr(),
          m->reset_input_grads[0],
          m->reset_input_grads[1]);
    } else {
      assert(false && "unsupport datatype in SigmoidSiluMulti");
    }
    if (m->profiling) {
      cudaEventRecord(t_end, stream);
      checkCUDA(cudaEventSynchronize(t_end));
      float elapsed = 0;
      checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
      cudaEventDestroy(t_start);
      cudaEventDestroy(t_end);
      printf("[SigmoidSiluMulti] peft_bwd time (CF) = %.9fms\n", elapsed);
    }
  }
  if (num_peft_requests == 0) {
    // No PEFT requests
    return;
  } else {
    // Otherwise assume at most 1 peft request
    assert(num_peft_requests == 1);
    assert(num_peft_tokens >= 1);
  }
  assert(false);
}

}; // namespace FlexFlow

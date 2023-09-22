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
    : OpMeta(handle) {
  profiling = ssm->profiling;
}

SigmoidSiluMultiMeta::~SigmoidSiluMultiMeta(void) {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
}

__device__ __forceinline__ float sigmoid_float(float x) {
  return 1.0 / (1.0 + expf(-x));
}

__device__ __forceinline__ half sigmoid_half(half x) {
  return (half)1.0 / ((half)1.0 + hexp(-x));
}

__global__ void SigmoidSiluMultiKernelFloat(int num_elements,
                                            float const *input1_ptr,
                                            float const *input2_ptr,
                                            float *output_ptr) {
  CUDA_KERNEL_LOOP(i, num_elements) {
    output_ptr[i] =
        input1_ptr[i] * sigmoid_float(input1_ptr[i]) * input2_ptr[i];
  }
}

__global__ void SigmoidSiluMultiKernelHalf(int num_elements,
                                           half const *input1_ptr,
                                           half const *input2_ptr,
                                           half *output_ptr) {
  CUDA_KERNEL_LOOP(i, num_elements) {
    output_ptr[i] = input1_ptr[i] * sigmoid_half(input1_ptr[i]) * input2_ptr[i];
  }
}

/*static*/
void SigmoidSiluMulti::inference_kernel_wrapper(
    SigmoidSiluMultiMeta const *m,
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
  if (m->input_type[0] == DT_FLOAT) {
    SigmoidSiluMultiKernelFloat<<<GET_BLOCKS(num_elements),
                                  min(CUDA_NUM_THREADS, num_elements),
                                  0,
                                  stream>>>(input1.domain.get_volume(),
                                            input1.get_float_ptr(),
                                            input2.get_float_ptr(),
                                            output.get_float_ptr());
  } else if (m->input_type[0] == DT_HALF) {
    SigmoidSiluMultiKernelHalf<<<GET_BLOCKS(num_elements),
                                 min(CUDA_NUM_THREADS, num_elements),
                                 0,
                                 stream>>>(input1.domain.get_volume(),
                                           input1.get_half_ptr(),
                                           input2.get_half_ptr(),
                                           output.get_half_ptr());
  } else {
    assert(false && "unsupport datatype in layernorm");
  }

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[SigmoidSiluMulti] forward time (CF) = %.9fms\n", elapsed);
    // if (m->input_type[0] == DT_FLOAT) {
    //   print_tensor<float>(input.get_float_ptr(),
    //                       32,
    //                       "[SigmoidSiluMulti:forward:input]");
    //   print_tensor<float>(attn_bias.get_float_ptr(),
    //                       32,
    //                       "[SigmoidSiluMulti:forward:attn_bias]");
    //   print_tensor<float>(residual.get_float_ptr(),
    //                       32,
    //                       "[SigmoidSiluMulti:forward:residual]");
    //   print_tensor<float>(added_output.get_float_ptr(),
    //                       32,
    //                       "[SigmoidSiluMulti:forward:added_output]");
    //   print_tensor<float>(output.get_float_ptr(),
    //                       32,
    //                       "[SigmoidSiluMulti:forward:output]");
    //   print_tensor<float>(gamma.get_float_ptr(),
    //                       32,
    //                       "[SigmoidSiluMulti:forward:gamma]");
    //   print_tensor<float>(
    //       beta.get_float_ptr(), 32,
    //       "[SigmoidSiluMulti:forward:beta]");
    // } else {
    //   print_tensor<half>(
    //       input.get_half_ptr(), 32,
    //       "[SigmoidSiluMulti:forward:input]");
    //   print_tensor<half>(attn_bias.get_half_ptr(),
    //                      32,
    //                      "[SigmoidSiluMulti:forward:attn_bias]");
    //   print_tensor<half>(residual.get_half_ptr(),
    //                      32,
    //                      "[SigmoidSiluMulti:forward:residual]");
    //   print_tensor<half>(added_output.get_half_ptr(),
    //                      32,
    //                      "[SigmoidSiluMulti:forward:added_output]");
    //   print_tensor<half>(output.get_half_ptr(),
    //                      32,
    //                      "[SigmoidSiluMulti:forward:output]");
    //   print_tensor<half>(
    //       gamma.get_half_ptr(), 32,
    //       "[SigmoidSiluMulti:forward:gamma]");
    //   print_tensor<half>(
    //       beta.get_half_ptr(), 32,
    //       "[SigmoidSiluMulti:forward:beta]");
    // }
    // print_tensor<T>(in_ptr, 32, "[SigmoidSiluMulti:forward:input]");
    // print_tensor<T>(out_ptr, 32,
    // "[SigmoidSiluMulti:forward:output]");
  }
}

}; // namespace FlexFlow

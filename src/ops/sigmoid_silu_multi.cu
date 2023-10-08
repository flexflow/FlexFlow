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
  inference_debugging = ssm->inference_debugging;
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

}; // namespace FlexFlow

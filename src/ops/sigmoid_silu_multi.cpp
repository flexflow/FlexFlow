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

#include "flexflow/ops/sigmoid_silu_multi.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

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
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  int num_elements = input1.domain.get_volume();
  assert(input2.domain.get_volume() == num_elements);
  assert(output.domain.get_volume() == num_elements);

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
  }

  if (m->input_type[0] == DT_FLOAT) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(SigmoidSiluMultiKernelFloat),
                       GET_BLOCKS(num_elements),
                       min(CUDA_NUM_THREADS, num_elements),
                       0,
                       stream,
                       input1.domain.get_volume(),
                       input1.get_float_ptr(),
                       input2.get_float_ptr(),
                       output.get_float_ptr());
  } else if (m->input_type[0] == DT_HALF) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(SigmoidSiluMultiKernelHalf),
                       GET_BLOCKS(num_elements),
                       min(CUDA_NUM_THREADS, num_elements),
                       0,
                       stream,
                       input1.domain.get_volume(),
                       input1.get_half_ptr(),
                       input2.get_half_ptr(),
                       output.get_half_ptr());
  } else {
    assert(false && "unsupport datatype in SigmoidSiluMulti");
  }

  if (m->profiling) {
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
    printf("[SigmoidSiluMulti] forward time (CF) = %.9fms\n", elapsed);
  }
}

}; // namespace FlexFlow

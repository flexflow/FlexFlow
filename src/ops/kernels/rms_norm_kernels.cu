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

#include "flexflow/ops/kernels/rms_norm_kernels.h"
#include "flexflow/ops/rms_norm.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;
using Legion::Domain;

RMSNormMeta::RMSNormMeta(FFHandler handler, RMSNorm const *rms)
    : OpMeta(handler, rms) {
  eps = rms->eps;
  // fixme
  checkCUDA(cudaMalloc(&mean_ptr, sizeof(float) * 1000));
}

namespace Kernels {
namespace RMSNorm {

void forward_kernel_wrapper(RMSNormMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorR const &weight,
                            GenericTensorAccessorW const &output
                            ) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }  

  Internal::forward_kernel(input.get_float_ptr(),
                             weight.get_float_ptr(),
                             output.get_float_ptr(),
                             input.domain.get_volume(),
                             stream);
   if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[RMSNorm] forward time (CF) = %.2fms\n", elapsed);
    print_tensor<float>(input.get_float_ptr(), 32, "[RMSNorm:forward:input]");
    print_tensor<float>(output.get_float_ptr(), 32, "[RMSNorm:forward:output]");
  }

}

namespace Internal {
/*static*/
void forward_kernel(float const *input_ptr,
                             float const *weight_ptr,
                             float *output_ptr,
                             coord_t dim_size,
                             cudaStream_t stream) {
    //pow

    //reduce

    //add eps

    //multiply with x


    //apply weights   

    return;                       

}
} // namespace Internal
} // namespace RMSNorm
} // namespace Kernels
} // namespace FlexFlow
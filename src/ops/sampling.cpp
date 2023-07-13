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

#include "flexflow/ops/sampling.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

/*static*/
template <typename DT>
void Sampling::forward_kernel(SamplingMeta const *m,
                              DT *input_ptr,
                              int *indices_ptr,
                              float const top_p,
                              int const length,
                              int const batch_size,
                              hipStream_t stream) {}

/*static*/
void Sampling::forward_kernel_wrapper(SamplingMeta const *m,
                                      GenericTensorAccessorW const &input,
                                      GenericTensorAccessorW const &indices) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    hipEventCreate(&t_start);
    hipEventCreate(&t_end);
    hipEventRecord(t_start, stream);
  }

  handle_unimplemented_hip_kernel(OP_RMS_NORM);

  if (m->profiling) {
    hipEventRecord(t_end, stream);
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    hipEventDestroy(t_start);
    hipEventDestroy(t_end);
  }
}

SamplingMeta::SamplingMeta(FFHandler handler,
                           Op const *op,
                           int batch_size,
                           int total_ele,
                           GenericTensorAccessorW input)
    : OpMeta(handler, op) {}

}; // namespace FlexFlow
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

#include "flexflow/ops/experts.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

/*static*/
void Experts::forward_kernel_wrapper(ExpertsMeta const *m,
                                     float const *input,
                                     int const *indices,
                                     float const *topk_gate_preds,
                                     float *output,
                                     int chosen_experts,
                                     int batch_size,
                                     int out_dim) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  int expert_capacity =
      ceil(m->alpha * chosen_experts / m->num_experts * batch_size);

  // TODO: write the HIP version of the kernel after finishing the CUDA kernel
}

ExpertsMeta::ExpertsMeta(FFHandler handler,
                         int _num_experts,
                         int _experts_start_idx,
                         float _alpha)
    : OpMeta(handler), num_experts(_num_experts),
      experts_start_idx(_experts_start_idx), alpha(_alpha) {}
ExpertsMeta::~ExpertsMeta(void) {}

}; // namespace FlexFlow

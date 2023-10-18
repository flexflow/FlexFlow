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
                                     float const *weights,
                                     float const *biases,
                                     int num_active_infr_tokens,
                                     int chosen_experts,
                                     int batch_size,
                                     int out_dim) {
  // TODO: write the HIP version of the kernel after finishing the CUDA kernel
  handle_unimplemented_hip_kernel(OP_EXPERTS);
}

ExpertsMeta::ExpertsMeta(FFHandler handler, Experts const *e)
    : OpMeta(handler, e), num_experts(e->num_experts),
      experts_start_idx(e->experts_start_idx), data_dim(e->data_dim),
      out_dim(e->out_dim), experts_num_layers(e->experts_num_layers),
      experts_internal_dim_size(e->experts_internal_dim_size),
      effective_batch_size(e->effective_batch_size),
      num_chosen_experts(e->num_chosen_experts), alpha(e->alpha),
      use_bias(e->use_bias), activation(e->activation) {}

ExpertsMeta::~ExpertsMeta(void) {}

}; // namespace FlexFlow

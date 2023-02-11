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
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

/*static*/
void Experts::forward_kernel_wrapper(ExpertsMeta const *m,
                                     float const *input,
                                     int const *indices,
                                     float const *topk_gate_preds,
                                     float *output,
                                     float const **weights,
                                     int chosen_experts,
                                     int batch_size,
                                     int out_dim) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  int expert_capacity =
      ceil(m->alpha * chosen_experts / m->num_experts * batch_size);

  int num_experts = m->num_experts;
  // int expert_start_index = experts_start_idx;
  bool use_bias = m->use_bias;
  // ActiMode activation = m->activation;

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  cudaMemcpy(m->dev_weights,
             weights,
             num_experts * (1 + use_bias) * sizeof(float *),
             cudaMemcpyHostToDevice);

  /** TODO: launch one or more kernel(s) to do the following:
   * 1. sort the tokens by expert to which they are assigned. This will require
   * replicating tokens when chosen_experts > 1
   * 2. matrix multiply (you can use cublasGemmEx) each slice of tokens with the
   * corresponding expert's weights tensor. Add the bias.
   *      - you can obtain the slice by selecting the tokens between the index
   * where the expert i starts and min(i+expert_capacity, index where expert i+1
   * starts)
   * 3. reorder the outputs by token, and aggregate the outputs of multiple
   * experts for the same token by computing an average weighted by the
   * appropriate coefficient from the topk_gate_preds matrix.
   */

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[Experts] forward time = %.2lfms\n", elapsed);
  }
}

ExpertsMeta::ExpertsMeta(FFHandler handler,
                         int _num_experts,
                         int _experts_start_idx,
                         float _alpha,
                         bool _use_bias,
                         ActiMode _activation)
    : OpMeta(handler), num_experts(_num_experts),
      experts_start_idx(_experts_start_idx), alpha(_alpha), use_bias(_use_bias),
      activation(_activation) {
  checkCUDA(
      cudaMalloc(&dev_weights, num_experts * (1 + use_bias) * sizeof(float *)));
}
ExpertsMeta::~ExpertsMeta(void) {
  checkCUDA(cudaFree(&dev_weights));
}

}; // namespace FlexFlow

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

__global__ void experts_forward_kernel(float const *input,
                                       int const *indices,
                                       float const *topk_gate_preds,
                                       float **outputs,
                                       int num_experts,
                                       int experts_start_idx,
                                       int chosen_experts,
                                       int expert_capacity,
                                       int batch_size,
                                       int out_dim) {
  // shared at the block level
  __shared__ float token_assigned[MAX_BATCH_SIZE][MAX_EXPERTS_PER_BLOCK];

  // initialize the token assignments to 0
  CUDA_KERNEL_LOOP(i, MAX_BATCH_SIZE * MAX_EXPERTS_PER_BLOCK) {
    int token_index = i / MAX_EXPERTS_PER_BLOCK;
    int expert_index = i % MAX_EXPERTS_PER_BLOCK;
    token_assigned[token_index][expert_index] = 0.0f;
  }

  __syncthreads();

  // Compute token assignments, single thread per block
  if (threadIdx.x == 0) {
    int token_count[MAX_EXPERTS_PER_BLOCK] = {0};
    for (int i = 0; i < chosen_experts * batch_size; i++) {
      // Get the token index, between 0 and batch_size
      int token_index = i / chosen_experts;
      // Get global index (indices[i]) of expert to which the token is assigned,
      // and compute the local index (expert_index) of the expert within the
      // block of fused experts
      int expert_index = indices[i] - experts_start_idx;
      // check if the token is assigned to an expert in this block, and if so,
      // whether the expert still has capacity not that since each expert is
      // assigned to only one block, it is safe to reason about expert capacity
      // locally
      if (expert_index >= 0 && expert_index < num_experts &&
          token_count[expert_index] < expert_capacity) {
        token_assigned[token_index][expert_index] = topk_gate_preds[i];
        token_count[expert_index]++;
      } else {
      }
    }
  }

  __syncthreads();

  // compute output
  CUDA_KERNEL_LOOP(i, num_experts * batch_size * out_dim) {
    // output indexing:
    // i = expert_index*(batch_size*out_dim) + token_index*out_dim + dim_index
    // input indexing:
    // i = token_index * (num_experts * out_dim) + expert_index * out_dim +
    // dim_index
    int expert_index = i / (batch_size * out_dim);
    // int token_index = (i - expert_index*(batch_size*out_dim)) / out_dim;
    int token_index = (i % (batch_size * out_dim)) / out_dim;
    // int dim_index = i - expert_index*(batch_size*out_dim) -
    // token_index*out_dim;
    int dim_index = i % out_dim;
    outputs[expert_index][token_index * out_dim + dim_index] =
        input[i] * token_assigned[token_index][expert_index];
  }
}

/*static*/
void Experts::forward_kernel_wrapper(ExpertsMeta const *m,
                                     float const *input,
                                     int const *indices,
                                     float const *topk_gate_preds,
                                     float **outputs,
                                     int num_experts,
                                     int experts_start_idx,
                                     int expert_capacity,
                                     int chosen_experts,
                                     int batch_size,
                                     int out_dim) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  //   cudaEvent_t t_start, t_end;
  //   if (m->profiling) {
  //     cudaEventCreate(&t_start);
  //     cudaEventCreate(&t_end);
  //     cudaEventRecord(t_start, stream);
  //   }
  hipMemcpy(m->dev_region_ptrs,
            outputs,
            num_experts * sizeof(float *),
            hipMemcpyHostToDevice);

  hipLaunchKernelGGL(
      experts_forward_kernel,
      GET_BLOCKS(batch_size * num_experts * out_dim),
      min(CUDA_NUM_THREADS, (int)(batch_size * num_experts * out_dim)),
      0,
      stream,
      input,
      indices,
      topk_gate_preds,
      m->dev_region_ptrs,
      num_experts,
      experts_start_idx,
      chosen_experts,
      expert_capacity,
      batch_size,
      out_dim);

  // if (m->profiling) {
  //     cudaEventRecord(t_end, stream);
  //     checkCUDA(cudaEventSynchronize(t_end));
  //     float elapsed = 0;
  //     checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //     cudaEventDestroy(t_start);
  //     cudaEventDestroy(t_end);
  //     printf("[Experts] forward time = %.2lfms\n", elapsed);
  // }
}

ExpertsMeta::ExpertsMeta(FFHandler handler, int num_experts) : OpMeta(handler) {
  checkCUDA(hipMalloc(&dev_region_ptrs, num_experts * sizeof(float *)));
}
ExpertsMeta::~ExpertsMeta(void) {
  checkCUDA(hipFree(&dev_region_ptrs));
}

}; // namespace FlexFlow

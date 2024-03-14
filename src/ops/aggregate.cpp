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

#include "flexflow/ops/aggregate.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

__global__ void agg_forward_kernel(float **exp_preds,
                                   int const *exp_assign,
                                   float const *gate_net_preds,
                                   float *output,
                                   int n,
                                   int const k,     // num chosen experts
                                   int exp_samples, // max samples per expert
                                   int const batch_size,
                                   int out_dim) {
  __shared__ float
      *chosen_exp_preds[AGGREGATE_MAX_K * AGGREGATE_MAX_BATCH_SIZE];

  // Get pred pointers, single thread pre block
  if (threadIdx.x == 0) {
    int expert_idx[AGGREGATE_MAX_N] = {0};
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < k; j++) {
        // Get pointer to chosen expert predictions
        int expert = exp_assign[i * k + j];
        if (expert_idx[expert] >= exp_samples) {
          // dropped sample
          chosen_exp_preds[i * k + j] = 0;
          continue;
        }
        chosen_exp_preds[i * k + j] =
            exp_preds[expert] + expert_idx[expert] * out_dim;
        expert_idx[expert]++;
      }
    }
  }

  // set output tensor to 0
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    output[i] = 0.0f;
  }

  __syncthreads();

  // compute output
  CUDA_KERNEL_LOOP(i, k * out_dim * batch_size) {
    if (chosen_exp_preds[i / out_dim] != 0) {
      float res = gate_net_preds[i / out_dim] *
                  chosen_exp_preds[i / out_dim][i % (out_dim)];
      int out_id = (i / (k * out_dim)) * out_dim + (i % out_dim);
      atomicAdd(output + out_id, res);
    }
  }
}

__device__ void agg_backward_kernel_gate(float const *output_grad,
                                         float *full_gate_grads,
                                         float **exp_preds,
                                         int const *expert_assign,
                                         bool const *cache_corr,
                                         int *expert_bal,
                                         float lambda_bal,
                                         int batch_size,
                                         int k,
                                         int n,
                                         int out_dim) {
  // gate gradient
  CUDA_KERNEL_LOOP(i, batch_size * k * out_dim) {
    if (exp_preds[i / out_dim] != 0 && cache_corr[i / (k * out_dim)]) {
      int out_id = (i / (k * out_dim)) * out_dim + (i % out_dim);
      float res = output_grad[out_id] * exp_preds[i / out_dim][i % out_dim];

      float *gate_grad_idx =
          full_gate_grads + (i / (out_dim * k)) * n +
          expert_assign[(i / (out_dim * k)) * k + (i / out_dim) % k];
      atomicAdd(gate_grad_idx, res);
    }
  }

  // balance term
  CUDA_KERNEL_LOOP(i, n * batch_size) {
    atomicAdd(full_gate_grads + i, lambda_bal * expert_bal[i % n]);
  }

  __syncthreads();

  // make 0 mean
  CUDA_KERNEL_LOOP(i, batch_size * n) {
    int start = (i / n) * n;
    float sub = -full_gate_grads[i] / n;
    for (int j = 0; j < n; j++) {
      atomicAdd(full_gate_grads + start + j, sub);
    }
  }
}

__device__ void agg_backward_kernel_exp(float const *output_grad,
                                        float const *gate_preds,
                                        float **exp_grads,
                                        int batch_size,
                                        int k,
                                        int out_dim) {
  // compute expert gradients
  CUDA_KERNEL_LOOP(i, k * out_dim * batch_size) {
    if (exp_grads[i / out_dim] != 0) {
      int out_id = (i / (k * out_dim)) * out_dim + (i % out_dim);
      exp_grads[i / out_dim][i % out_dim] +=
          gate_preds[i / out_dim] * output_grad[out_id];
    }
  }
}

__global__ void agg_backward_kernel(float **exp_preds,
                                    float **exp_grads,
                                    int const *exp_assign,
                                    int const *true_exp_assign,
                                    float const *gating_net_preds,
                                    float *full_gating_grads,
                                    float const *output_grads,
                                    int n,           // num experts
                                    int k,           // num chosen experts
                                    int exp_samples, // max samples per expert
                                    float lambda_bal,
                                    int batch_size,
                                    int out_dim) {
  __shared__ float
      *chosen_exp_preds[AGGREGATE_MAX_K * AGGREGATE_MAX_BATCH_SIZE];
  __shared__ float
      *chosen_exp_grads[AGGREGATE_MAX_K * AGGREGATE_MAX_BATCH_SIZE];
  __shared__ int expert_bal[AGGREGATE_MAX_N];
  __shared__ bool cache_corr[AGGREGATE_MAX_BATCH_SIZE];

  // Get pred pointers, single thread per block
  if (threadIdx.x == 0) {
    // init arrays
    for (int i = 0; i < n; i++) {
      expert_bal[i] = 0;
    }
    for (int i = 0; i < batch_size; i++) {
      cache_corr[i] = true;
    }

    // Get pointer to chosen expert predictions and expert counts
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < k; j++) {
        int expert = true_exp_assign[k * i + j];
        if (expert != exp_assign[k * i + j]) {
          cache_corr[i] = false;
        }
        if (expert_bal[expert] >= exp_samples) {
          // dropped sample
          chosen_exp_preds[i * k + j] = 0;
          chosen_exp_grads[i * k + j] = 0;
          expert_bal[expert]++;
          continue;
        }
        chosen_exp_preds[i * k + j] =
            exp_preds[expert] + expert_bal[expert] * out_dim;
        chosen_exp_grads[i * k + j] =
            exp_grads[expert] + expert_bal[expert] * out_dim;
        expert_bal[expert]++;
      }
    }
  }

  __syncthreads();

  // FIXME: These 2 functions could execute independently in parallel
  // get expert gradients
  agg_backward_kernel_exp(
      output_grads, gating_net_preds, chosen_exp_grads, batch_size, k, out_dim);

  // get gating net gradients
  agg_backward_kernel_gate(output_grads,
                           full_gating_grads,
                           chosen_exp_preds,
                           exp_assign,
                           cache_corr,
                           expert_bal,
                           (lambda_bal * n) / batch_size,
                           batch_size,
                           k,
                           n,
                           out_dim);
}

/*static*/
void Aggregate::forward_kernel_wrapper(AggregateMeta const *m,
                                       float **exp_preds,
                                       int const *acc_gate_assign_ptr,
                                       float const *acc_gate_pred_ptr,
                                       float *acc_output_ptr,
                                       int n,
                                       int const k,
                                       int rows,
                                       int const batch_size,
                                       int out_dim) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDA(hipblasSetStream(m->handle.blas, stream));
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

  // call forward_kernel
  checkCUDA(hipMemcpy(
      m->dev_exp_preds, exp_preds, n * sizeof(float *), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(agg_forward_kernel,
                     GET_BLOCKS(batch_size * k * out_dim),
                     min(CUDA_NUM_THREADS, (int)(batch_size * k * out_dim)),
                     0,
                     stream,
                     m->dev_exp_preds,
                     acc_gate_assign_ptr,
                     acc_gate_pred_ptr,
                     acc_output_ptr,
                     n,
                     k,
                     rows,
                     batch_size,
                     out_dim);
}

/*static*/
void Aggregate::backward_kernel_wrapper(AggregateMeta const *m,
                                        float **exp_preds,
                                        float **exp_grads,
                                        int const *acc_gate_assign_ptr,
                                        int const *acc_true_gate_assign_ptr,
                                        float const *acc_gate_pred_ptr,
                                        float *full_acc_gate_grad_ptr,
                                        float const *acc_output_grad_ptr,
                                        int n,
                                        int const k,
                                        int rows,
                                        float lambda_bal,
                                        int const batch_size,
                                        int out_dim) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDA(hipblasSetStream(m->handle.blas, stream));
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

  // call backward kernel
  checkCUDA(hipMemcpy(
      m->dev_exp_preds, exp_preds, n * sizeof(float *), hipMemcpyHostToDevice));
  checkCUDA(hipMemcpy(
      m->dev_exp_grads, exp_grads, n * sizeof(float *), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(agg_backward_kernel,
                     GET_BLOCKS(batch_size * k * out_dim),
                     min(CUDA_NUM_THREADS, (int)(batch_size * k * out_dim)),
                     0,
                     stream,
                     m->dev_exp_preds,
                     m->dev_exp_grads,
                     acc_gate_assign_ptr,
                     acc_true_gate_assign_ptr,
                     acc_gate_pred_ptr,
                     full_acc_gate_grad_ptr,
                     acc_output_grad_ptr,
                     n,
                     k,
                     rows,
                     lambda_bal,
                     batch_size,
                     out_dim);
}

AggregateMeta::AggregateMeta(FFHandler handler, int n) : OpMeta(handler) {
  checkCUDA(hipMalloc(&dev_exp_preds, n * sizeof(float *)));
  checkCUDA(hipMalloc(&dev_exp_grads, n * sizeof(float *)));
}
AggregateMeta::~AggregateMeta(void) {
  checkCUDA(hipFree(&dev_exp_preds));
  checkCUDA(hipFree(&dev_exp_grads));
}

}; // namespace FlexFlow
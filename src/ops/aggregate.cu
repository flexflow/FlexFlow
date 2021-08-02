/* Copyright 2019 Stanford
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
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::Rect;
using Legion::PhysicalRegion;
using Legion::coord_t;

#define MAX_K 4
#define MAX_BATCH_SIZE 32
#define MAX_N 12

OpMeta* Aggregate::init_task(const Task* task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime* runtime)
{
  Aggregate* agg = (Aggregate*) task->args;
  FFHandler handle = *((FFHandler*)task->local_args);
  AggregateMeta* m = new AggregateMeta(handle, agg->n);
  m->profiling = agg->profiling;
  return m;
}

__global__
void agg_forward_kernel(float** exp_preds,
        const int* exp_assign,
        const float* gate_net_preds,
        float* output,
        int n,
        const int k, // num chosen experts
        int exp_samples, // max samples per expert
        const int batch_size,
        int out_dim)
{
  __shared__ float* chosen_exp_preds[MAX_K*MAX_BATCH_SIZE];

  // Get pred pointers, single thread pre block
  if(threadIdx.x == 0) {
    int expert_idx[MAX_N] = {0};
    for(int i = 0; i < batch_size; i++) {
      for(int j = 0; j < k; j++) {
        // Get pointer to chosen expert predictions
        int expert = exp_assign[i*k+j];
        if(expert_idx[expert] >= exp_samples) {
          // dropped sample
          chosen_exp_preds[i*k+j] = 0;
          continue;
        }
        chosen_exp_preds[i*k+j] = exp_preds[expert] + expert_idx[expert]*out_dim;
        expert_idx[expert]++;
      }
    }
  }

  // set output tensor to 0
  CUDA_KERNEL_LOOP(i, batch_size*out_dim)
  {
    output[i] = 0.0f;
  }

  __syncthreads();

  // compute output
  CUDA_KERNEL_LOOP(i, k*out_dim*batch_size)
  {
    if(chosen_exp_preds[i/out_dim] != 0) {
      float res = gate_net_preds[i/out_dim] * chosen_exp_preds[i/out_dim][i%(out_dim)];
      int out_id = (i/(k*out_dim))*out_dim + (i%out_dim);
      atomicAdd(output+out_id, res);
    }
  }
}


__device__
void agg_backward_kernel_gate(const float* output_grad,
              float* full_gate_grads,
              float** exp_preds,
              const int* expert_assign,
              const bool* cache_corr,
              int* expert_bal, float lambda_bal,
              int batch_size, int k, int n, int out_dim)
{
  // gate gradient
  CUDA_KERNEL_LOOP(i, batch_size*k*out_dim)
  {
    if (exp_preds[i/out_dim] != 0 && cache_corr[i/(k*out_dim)]) {
      int out_id = (i/(k*out_dim))*out_dim + (i%out_dim);
      float res = output_grad[out_id] * exp_preds[i/out_dim][i%out_dim];

      float* gate_grad_idx = full_gate_grads + (i/(out_dim*k))*n
        + expert_assign[(i/(out_dim*k))*k+(i/out_dim)%k];
      atomicAdd(gate_grad_idx, res);
    }
  }

  // balance term
  CUDA_KERNEL_LOOP(i, n*batch_size)
  {
    atomicAdd(full_gate_grads+i, lambda_bal*expert_bal[i%n]);
  }

  __syncthreads();

  // make 0 mean
  CUDA_KERNEL_LOOP(i, batch_size*n)
  {
    int start = (i/n)*n;
    float sub = -full_gate_grads[i]/n;
    for(int j = 0; j < n; j++) {
      atomicAdd(full_gate_grads+start+j, sub);
    }
  }
}


__device__
void agg_backward_kernel_exp(const float* output_grad,
              const float* gate_preds,
              float** exp_grads,
              int batch_size,
              int k,
              int out_dim) {
  // compute expert gradients
  CUDA_KERNEL_LOOP(i, k*out_dim*batch_size)
  {
    if (exp_grads[i/out_dim] != 0) {
      int out_id = (i/(k*out_dim))*out_dim + (i%out_dim);
      exp_grads[i/out_dim][i%out_dim] += gate_preds[i/out_dim] * output_grad[out_id];
    }
  }
}


__global__
void agg_backward_kernel(float** exp_preds,
        float** exp_grads,
        const int* exp_assign,
        const int* true_exp_assign,
        const float* gating_net_preds,
        float* full_gating_grads,
        const float* output_grads,
        int n, // num experts
        int k, // num chosen experts
        int exp_samples, // max samples per expert
        float lambda_bal,
        int batch_size,
        int out_dim)
{
  __shared__ float* chosen_exp_preds[MAX_K*MAX_BATCH_SIZE];
  __shared__ float* chosen_exp_grads[MAX_K*MAX_BATCH_SIZE];
  __shared__ int expert_bal[MAX_N];
  __shared__ bool cache_corr[MAX_BATCH_SIZE];

  // Get pred pointers, single thread per block
  if(threadIdx.x == 0) {
    // init arrays
    for(int i = 0; i < n; i++) expert_bal[i] = 0;
    for(int i = 0; i < batch_size; i++) cache_corr[i] = true;

    // Get pointer to chosen expert predictions and expert counts
    for(int i = 0; i < batch_size; i++) {
      for(int j = 0; j < k; j++) {
        int expert = true_exp_assign[k*i + j];
        if(expert != exp_assign[k*i + j])
          cache_corr[i] = false;
        if(expert_bal[expert] >= exp_samples) {
          // dropped sample
          chosen_exp_preds[i*k+j] = 0;
          chosen_exp_grads[i*k+j] = 0;
          expert_bal[expert]++;
          continue;
        }
        chosen_exp_preds[i*k+j] = exp_preds[expert] + expert_bal[expert]*out_dim;
        chosen_exp_grads[i*k+j] = exp_grads[expert] + expert_bal[expert]*out_dim;
        expert_bal[expert]++;
      }
    }
  }

  __syncthreads();

  // FIXME: These 2 functions could execute independently in parallel
  // get expert gradients
  agg_backward_kernel_exp(output_grads, gating_net_preds, chosen_exp_grads,
    batch_size, k, out_dim);

  // get gating net gradients
  agg_backward_kernel_gate(output_grads, full_gating_grads, chosen_exp_preds,
    exp_assign, cache_corr, expert_bal, (lambda_bal*n)/batch_size, batch_size,
    k, n, out_dim);
}


void Aggregate::forward_task(const Task *task,
                             const std::vector<PhysicalRegion>& regions,
                             Context ctx, Runtime* runtime)
{
  int n = ((Aggregate*)task->args)->n;

  assert((int)regions.size() == n+3);
  assert((int)task->regions.size() == n+3);

  const AggregateMeta* m = *((AggregateMeta**)task->local_args);

  // get gate_pred, gate_assign, output
  const AccessorRO<float, 2> acc_gate_pred(regions[0], FID_DATA);
  const AccessorRO<int, 2> acc_gate_assign(regions[1], FID_DATA);
  const AccessorWO<float, 2> acc_output(regions[n+2], FID_DATA);

  Rect<2> rect_gate_pred = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_gate_assign = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_output = runtime->get_index_space_domain(
      ctx, task->regions[n+2].region.get_index_space());

  coord_t batch_size = rect_gate_pred.hi[1] - rect_gate_pred.lo[1] + 1;
  assert(batch_size == rect_gate_assign.hi[1] - rect_gate_assign.lo[1] + 1);
  assert(rect_gate_pred.hi[0] - rect_gate_pred.lo[0] == rect_gate_assign.hi[0] - rect_gate_assign.lo[0]);
  assert(batch_size == rect_output.hi[1] - rect_output.lo[1] + 1);
  coord_t out_dim = rect_output.hi[0] - rect_output.lo[0] + 1;

  // get exp_preds
  float* exp_preds[n];
  // get first exp_pred and row and out_dim
  Domain exp_domain = runtime->get_index_space_domain(
    ctx, task->regions[2].region.get_index_space());
  exp_preds[0] = helperGetTensorPointerWO<float>(
    regions[2], task->regions[2], FID_DATA, ctx, runtime);
  coord_t rows = exp_domain.hi()[1] - exp_domain.lo()[1] + 1;
  assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);

  for(int i = 1; i < n; i++) {
    exp_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+2].region.get_index_space());
    exp_preds[i] = helperGetTensorPointerWO<float>(
      regions[i+2], task->regions[i+2], FID_DATA, ctx, runtime);

    assert(rows == exp_domain.hi()[1] - exp_domain.lo()[1] + 1);
    assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);
  }

  int k = (int)(rect_gate_assign.hi[0] - rect_gate_assign.lo[0] + 1);

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  // call forward_kernel
  cudaMemcpy(m->dev_exp_preds, exp_preds, n*sizeof(float*), cudaMemcpyHostToDevice);

  agg_forward_kernel<<<GET_BLOCKS(batch_size*k*out_dim), min(CUDA_NUM_THREADS,(int)(batch_size*k*out_dim)), 0, stream>>>(
    m->dev_exp_preds, acc_gate_assign.ptr(rect_gate_assign), acc_gate_pred.ptr(rect_gate_pred),
    acc_output.ptr(rect_output), n, k, rows, batch_size, out_dim);
}


void Aggregate::backward_task(const Task *task,
                              const std::vector<PhysicalRegion>& regions,
                              Context ctx, Runtime* runtime)
{
  const AggregateMeta* m = *((AggregateMeta**)task->local_args);
  int n = ((Aggregate*)task->args)->n;
  float lambda_bal = ((Aggregate*)task->args)->lambda_bal;

  assert((int)regions.size() == 2*n+5);
  assert((int)task->regions.size() == 2*n+5);

  // get gate_pred, gate_grad, gate_assign, output_grad
  const AccessorRO<float, 2> acc_gate_pred(regions[0], FID_DATA);
  const AccessorRO<int, 2> acc_gate_assign(regions[1], FID_DATA);
  const AccessorRO<int, 2> acc_true_gate_assign(regions[2], FID_DATA);
  const AccessorWO<float, 2> full_acc_gate_grad(regions[3], FID_DATA);
  const AccessorRO<float, 2> acc_output_grad(regions[2*n+4], FID_DATA);

  Rect<2> rect_gate_pred = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_gate_assign = runtime->get_index_space_domain(
       ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_true_gate_assign = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  Rect<2> rect_full_gate_grad = runtime->get_index_space_domain(
      ctx, task->regions[3].region.get_index_space());
  Rect<2> rect_out_grad = runtime->get_index_space_domain(
      ctx, task->regions[2*n+4].region.get_index_space());

  coord_t batch_size = rect_gate_pred.hi[1] - rect_gate_pred.lo[1] + 1;
  assert(batch_size == rect_gate_assign.hi[1] - rect_gate_assign.lo[1] + 1);
  assert(rect_gate_assign == rect_true_gate_assign);
  assert(batch_size == rect_out_grad.hi[1] - rect_out_grad.lo[1] + 1);
  assert(batch_size == rect_full_gate_grad.hi[1] - rect_full_gate_grad.lo[1] + 1);
  coord_t k = rect_gate_assign.hi[0] - rect_gate_assign.lo[0] + 1;
  assert(rect_gate_pred.hi[0] - rect_gate_pred.lo[0] + 1 == k);
  coord_t out_dim = rect_out_grad.hi[0] - rect_out_grad.lo[0] + 1;
  assert(n == rect_full_gate_grad.hi[0] - rect_full_gate_grad.lo[0] + 1);

  // get exp_preds
  float* exp_preds[n];
  // get first exp_pred and row
  Domain exp_domain = runtime->get_index_space_domain(
    ctx, task->regions[4].region.get_index_space());
  exp_preds[0] = helperGetTensorPointerRW<float>(
    regions[4], task->regions[4], FID_DATA, ctx, runtime);
  coord_t rows = exp_domain.hi()[1] - exp_domain.lo()[1] + 1;
  assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);

  for(int i = 1; i < n; i++) {
    exp_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+4].region.get_index_space());
    exp_preds[i] = helperGetTensorPointerRW<float>(
      regions[i+4], task->regions[i+4], FID_DATA, ctx, runtime);
    assert(rows == exp_domain.hi()[1] - exp_domain.lo()[1] + 1);
    assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);
  }

  // get chosen_exp_grads
  float* exp_grads[n];
  for(int i = 0; i < n; i++) {
    exp_domain = runtime->get_index_space_domain(
      ctx, task->regions[n+i+4].region.get_index_space());
    exp_grads[i] = helperGetTensorPointerRW<float>(
      regions[n+i+4], task->regions[n+i+4], FID_DATA, ctx, runtime);
    assert(rows == exp_domain.hi()[1] - exp_domain.lo()[1] + 1);
    assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);
  }

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  // call backward kernel
  cudaMemcpy(m->dev_exp_preds, exp_preds, n*sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(m->dev_exp_grads, exp_grads, n*sizeof(float*), cudaMemcpyHostToDevice);

  agg_backward_kernel<<<GET_BLOCKS(batch_size*k*out_dim), min(CUDA_NUM_THREADS,(int)(batch_size*k*out_dim)), 0, stream>>>(
    m->dev_exp_preds, m->dev_exp_grads, acc_gate_assign.ptr(rect_gate_assign),
    acc_true_gate_assign.ptr(rect_true_gate_assign), acc_gate_pred.ptr(rect_gate_pred),
    full_acc_gate_grad.ptr(rect_full_gate_grad), acc_output_grad.ptr(rect_out_grad),
    n, k, rows, lambda_bal, batch_size, out_dim);
}

AggregateMeta::AggregateMeta(FFHandler handler, int n)
: OpMeta(handler)
{
  checkCUDA(cudaMalloc(&dev_exp_preds, n*sizeof(float*)));
  checkCUDA(cudaMalloc(&dev_exp_grads, n*sizeof(float*)));
}
AggregateMeta::~AggregateMeta(void)
{
  checkCUDA(cudaFree(&dev_exp_preds));
  checkCUDA(cudaFree(&dev_exp_grads));
}


bool Aggregate::measure_operator_cost(Simulator* sim,
                                 const ParallelConfig& pc,
                                 CostMetrics& cost_metrics) const
{
  //TODO: implement
  cost_metrics.forward_time = 0.0f;
  cost_metrics.backward_time = 0.0f;
  cost_metrics.memory_requirement = 0;
  return false;
}

}; // namespace FlexFlow
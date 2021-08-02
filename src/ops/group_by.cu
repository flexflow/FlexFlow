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

#include "flexflow/ops/groupby.h"
#include "flexflow/utils/cuda_helper.h"
#include <math.h>
#include <stdio.h>

#define MAX_K 4
#define MAX_BATCH_SIZE 32
#define MAX_N 12

namespace FlexFlow {
// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::Rect;
using Legion::PhysicalRegion;
using Legion::coord_t;

OpMeta* Group_by::init_task(const Task* task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime* runtime)
{
  Group_by* gb = (Group_by*) task->args;
  FFHandler handle = *((FFHandler*)task->local_args);
  GroupByMeta* m = new GroupByMeta(handle, gb->n);
  m->profiling = gb->profiling;
  return m;
}

__global__
void gb_forward_kernel(const float* input,
        const int* exp_assign,
        float** outputs,
        int n, // num experts
        int k, // chosen experts
        float alpha, // factor additional memory assigned
        int batch_size,
        int data_dim)
{
  __shared__ float* chosen_exp_preds[MAX_K*MAX_BATCH_SIZE];

  // Get pred pointers, single thread per block
  if(threadIdx.x == 0) {
    int exp_tensor_rows = ceil(alpha*k/n*batch_size);
    int expert_idx[MAX_N] = {0};
    for(int i = 0; i < k*batch_size; i++) {
      // Get pointer to chosen expert predictions
      int expert = exp_assign[i];
      if(expert_idx[expert] >= exp_tensor_rows) {
        // dropped sample
        chosen_exp_preds[i] = 0;
        continue;
      }
      chosen_exp_preds[i] = outputs[expert] + expert_idx[expert]*data_dim;
      expert_idx[expert]++;
    }
  }

  __syncthreads();

  // compute output
  CUDA_KERNEL_LOOP(i, k*batch_size*data_dim)
  {
    if(chosen_exp_preds[i/data_dim] != 0) {
      float a = input[(i/(k*data_dim))*data_dim + i%data_dim];
      chosen_exp_preds[i/data_dim][i%data_dim] = a;
    }
  }
}


__global__
void gb_backward_kernel(float* input_grad,
        const int* exp_assign,
        float** output_grads,
        int n, // num experts
        int k, // chosen experts
        float alpha, // factor additional memory assigned
        int batch_size,
        int data_dim)
{
  __shared__ float* chosen_exp_grads[MAX_K*MAX_BATCH_SIZE];

  // Get pred pointers, single thread
  if(blockIdx.x * blockDim.x + threadIdx.x == 0) {
    int exp_tensor_rows = ceil(alpha*k/n*batch_size);
    int expert_idx[MAX_N] = {0};
    for(int i = 0; i < k*batch_size; i++) {
      // Get pointer to chosen expert predictions
      int expert = exp_assign[i];
      if(expert_idx[expert] >= exp_tensor_rows) {
        // dropped sample
        chosen_exp_grads[i] = 0;
        continue;
      }
      chosen_exp_grads[i] = output_grads[expert] + expert_idx[expert]*data_dim;
      expert_idx[expert]++;
    }
  }

  __syncthreads();

  // compute output
  CUDA_KERNEL_LOOP(i, k*batch_size*data_dim)
  {
    if(chosen_exp_grads[i/data_dim] != 0) {
      input_grad[(i/(k*data_dim))*data_dim + i%data_dim] = chosen_exp_grads[i/data_dim][i%data_dim];
    }
  }
}


void Group_by::forward_task(const Task *task,
                            const std::vector<PhysicalRegion>& regions,
                            Context ctx, Runtime* runtime)
{
  // Get n, alpha
  const Group_by* gb = (Group_by*) task->args;
  int n = gb->n;
  float alpha = gb->alpha;

  assert((int)regions.size() == n+2);
  assert((int)task->regions.size() == n+2);

  const GroupByMeta* m = *((GroupByMeta**)task->local_args);

  // get input and assign regions
  const AccessorRO<float, 2> acc_input(regions[0], FID_DATA);
  const AccessorRO<int, 2> acc_assign(regions[1], FID_DATA);

  Rect<2> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_assign = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());

  coord_t input_rows = rect_input.hi[1] - rect_input.lo[1] + 1;
  coord_t input_cols = rect_input.hi[0] - rect_input.lo[0] + 1;
  assert(input_rows == rect_assign.hi[1] - rect_assign.lo[1] + 1);
  int k = rect_assign.hi[0] - rect_assign.lo[0] + 1;
  int batch_size = input_rows;
  int data_dim = input_cols;

  // get output
  float* outputs[n];
  //int exp_output_rows = (int)ceil(alpha*k/n*batch_size);
  for(int i = 0; i < n; i++) {
    Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+2].region.get_index_space());
    outputs[i] = helperGetTensorPointerWO<float>(
      regions[i+2], task->regions[i+2], FID_DATA, ctx, runtime);

    //coord_t output_rows = out_domain.hi()[1] - out_domain.lo()[1] + 1;
    coord_t output_cols = out_domain.hi()[0] - out_domain.lo()[0] + 1;
    //assert((int)output_rows == exp_output_rows);
    assert(output_cols == input_cols);
  }

  // TODO: why cublas/cudnn stream is needed here?
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  // call forward kernel
  cudaMemcpy(m->dev_region_ptrs, outputs, n*sizeof(float*), cudaMemcpyHostToDevice);

  gb_forward_kernel<<<GET_BLOCKS(batch_size*k*data_dim), min(CUDA_NUM_THREADS,(int)(batch_size*k*data_dim)), 0, stream>>>(
    acc_input.ptr(rect_input), acc_assign.ptr(rect_assign), m->dev_region_ptrs, n, k,
    alpha, batch_size, data_dim);
}


void Group_by::backward_task(const Task *task,
                            const std::vector<PhysicalRegion>& regions,
                            Context ctx, Runtime* runtime)
{
  // Get n, alpha
  const GroupByMeta* m = *((GroupByMeta**)task->local_args);
  const Group_by* gb = (Group_by*) task->args;
  int n = gb->n;
  float alpha = gb->alpha;

  assert((int)regions.size() == n+2);
  assert((int)task->regions.size() == n+2);

  // get input and assign regions
  const AccessorWO<float, 2> acc_input_grad(regions[0], FID_DATA);
  const AccessorRO<int, 2> acc_assign(regions[1], FID_DATA);

  Rect<2> rect_input_grad = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_assign = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());

  coord_t input_rows = rect_input_grad.hi[1] - rect_input_grad.lo[1] + 1;
  coord_t input_cols = rect_input_grad.hi[0] - rect_input_grad.lo[0] + 1;
  assert(input_rows == rect_assign.hi[1] - rect_assign.lo[1] + 1);
  int k = rect_assign.hi[0] - rect_assign.lo[0] + 1;
  int batch_size = input_rows;
  int data_dim = input_cols;

  // get output
  float* output_grads[n];
  //int exp_output_rows = (int)ceil(alpha*k/n*batch_size);
  for(int i = 0; i < n; i++) {
    Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+2].region.get_index_space());
    output_grads[i] = helperGetTensorPointerRW<float>(
      regions[i+2], task->regions[i+2], FID_DATA, ctx, runtime);

    //coord_t output_rows = out_domain.hi()[1] - out_domain.lo()[1] + 1;
    coord_t output_cols = out_domain.hi()[0] - out_domain.lo()[0] + 1;
    //assert((int)output_rows == exp_output_rows);
    assert(output_cols == input_cols);
  }

  // TODO: why cublas/cudnn stream is needed here
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  // call forward kernel
  cudaMemcpy(m->dev_region_ptrs, output_grads, n*sizeof(float*), cudaMemcpyHostToDevice);

  gb_backward_kernel<<<GET_BLOCKS(batch_size*k*data_dim), min(CUDA_NUM_THREADS,(int)(batch_size*k*data_dim)), 0, stream>>>(
    acc_input_grad.ptr(rect_input_grad), acc_assign.ptr(rect_assign), m->dev_region_ptrs,
    n, k, alpha, batch_size, data_dim);
}

GroupByMeta::GroupByMeta(FFHandler handler, int n)
: OpMeta(handler)
{
  checkCUDA(cudaMalloc(&dev_region_ptrs, n*sizeof(float*)));
}
GroupByMeta::~GroupByMeta(void)
{
  checkCUDA(cudaFree(&dev_region_ptrs));
}

bool Group_by::measure_operator_cost(Simulator* sim,
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

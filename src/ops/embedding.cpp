/* Copyright 2020 Stanford
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

#include <hip/hip_runtime.h>
#include "flexflow/ops/embedding.h"
#include "flexflow/utils/hip_helper.h"

namespace FlexFlow {
// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::Rect;
using Legion::PhysicalRegion;
using Legion::coord_t;

__host__
OpMeta* Embedding::init_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime* runtime)
{
  const Embedding* embed = (Embedding*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  EmbeddingMeta* m = new EmbeddingMeta(handle);
  m->profiling = embed->profiling;
  m->aggr = embed->aggr;
  return m;
}

__global__
void embed_forward(const int64_t* input,
                   float* output,
                   const float* embed,
                   int out_dim,
                   int in_dim,
                   int batch_size,
                   AggrMode aggr)
{
  CUDA_KERNEL_LOOP(i, batch_size * out_dim)
  {
    output[i] = 0;
    int idx = i / out_dim;
    int off = i % out_dim;
    for (int j = 0; j < in_dim; j++) {
      int64_t wordIdx = input[idx * in_dim + j];
      output[i] += embed[wordIdx * out_dim + off];
      if (aggr == AGGR_MODE_SUM) {
      } else {
        assert(aggr == AGGR_MODE_AVG);
        output[i] /= in_dim;
      }
    }
  }
}

__global__
void embed_backward(const int64_t* input,
                    const float* output,
                    float* embed,
                    int out_dim,
                    int in_dim,
                    int batch_size,
                    AggrMode aggr)
{
  CUDA_KERNEL_LOOP(i, batch_size * out_dim)
  {
    int idx = i / out_dim;
    int off = i % out_dim;
    float gradient;
    if (aggr == AGGR_MODE_SUM) {
       gradient = output[i];
    } else {
      assert(aggr == AGGR_MODE_AVG);
      gradient = output[i] / in_dim;
    }
    for (int j = 0; j < in_dim; j++) {
      int64_t wordIdx = input[idx * in_dim + j];
      atomicAdd(embed + wordIdx * out_dim + off, gradient);
    }
  }
}

void Embedding::forward_kernel(int64_t const *input_ptr,
                               float *output_ptr,
                               float const *weight_ptr,
                               int in_dim,
                               int out_dim,
                               int batch_size,
                               AggrMode aggr,
                               int outputSize,
                               hipStream_t stream)
{
  hipLaunchKernelGGL(embed_forward, GET_BLOCKS(outputSize), CUDA_NUM_THREADS, 0, stream, 
      input_ptr, output_ptr, weight_ptr, out_dim, in_dim, batch_size, aggr);
}

/*
  regions[0](I): input
  regions[1](O): output
  regions[2](I): kernel
*/
__host__
void Embedding::forward_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime* runtime)
{
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  switch (in_domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
      return forward_task_with_dim<DIM>(task, regions, ctx, runtime);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

template<int NDIM>
__host__
void Embedding::forward_task_with_dim(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  //const Embedding* embed = (Embedding*) task->args;
  const EmbeddingMeta* m = *((EmbeddingMeta**) task->local_args);
  TensorAccessorR<int64_t, NDIM> accInput(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, NDIM> accOutput(
      regions[1], task->regions[1], FID_DATA, ctx, runtime, false/*readOutput*/);
  TensorAccessorR<float, NDIM> accWeight(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  // Input matches Output
  for (int i = 1; i < NDIM; i++) {
    assert(accInput.rect.hi[i] == accOutput.rect.hi[i]);
    assert(accInput.rect.lo[i] == accOutput.rect.lo[i]);
  }
  // Weight matches Output
  assert(accWeight.rect.hi[0] - accWeight.rect.lo[0]
      == accOutput.rect.hi[0] - accOutput.rect.lo[0]);
  int in_dim = accInput.rect.hi[0] - accInput.rect.lo[0] + 1;
  int out_dim = accOutput.rect.hi[0] - accOutput.rect.lo[0] + 1;
  int batch_size = accOutput.rect.volume() / out_dim;
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  forward_kernel(accInput.ptr, accOutput.ptr, accWeight.ptr, in_dim, out_dim, batch_size,  m->aggr, accOutput.rect.volume(), stream);
  if (m->profiling) {
    checkCUDA(hipDeviceSynchronize());
    print_tensor<int64_t>(accInput.ptr, accInput.rect.volume(), "[Embedding:forward:input]");
    print_tensor<float>(accWeight.ptr, accWeight.rect.volume(), "[Embedding:forward:weight]");
    print_tensor<float>(accOutput.ptr, accOutput.rect.volume(), "[Embedding:forward:output]");
  }
}

void Embedding::backward_kernel(int64_t const *input_ptr,
                                float const *output_ptr,
                                float *weight_grad_ptr,
                                int in_dim,
                                int out_dim,
                                int batch_size,
                                AggrMode aggr,
                                int outputSize,
                                hipStream_t stream)
{
  hipLaunchKernelGGL(embed_backward, GET_BLOCKS(outputSize), CUDA_NUM_THREADS, 0, stream, 
      input_ptr, output_ptr, weight_grad_ptr, out_dim, in_dim, batch_size, aggr);
}

void Embedding::backward_task(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime *runtime)
{
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  switch (in_domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
      return backward_task_with_dim<DIM>(task, regions, ctx, runtime);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

template<int NDIM>
__host__
void Embedding::backward_task_with_dim(const Task *task,
                                       const std::vector<PhysicalRegion> &regions,
                                       Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  //const Embedding* embed = (Embedding*) task->args;
  const EmbeddingMeta* m = *((EmbeddingMeta**) task->local_args);
  TensorAccessorR<int64_t, NDIM> accInput(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorR<float, NDIM> accOutput(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  TensorAccessorW<float, NDIM> accWeightGrad(
      regions[2], task->regions[2], FID_DATA, ctx, runtime, true/*readOutput*/);
  // Input matches Output
  for (int i = 1; i < NDIM; i++) {
    assert(accInput.rect.hi[i] == accOutput.rect.hi[i]);
    assert(accInput.rect.lo[i] == accOutput.rect.lo[i]);
  }
  // WeightGrad matches Output
  assert(accWeightGrad.rect.hi[0] - accWeightGrad.rect.lo[0] == accOutput.rect.hi[0] - accOutput.rect.lo[0]);
  int in_dim = accInput.rect.hi[0] - accInput.rect.lo[0] + 1;
  int out_dim = accOutput.rect.hi[0] - accOutput.rect.lo[0] + 1;
  int batch_size = accOutput.rect.volume() / out_dim;
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  backward_kernel(accInput.ptr, accOutput.ptr, accWeightGrad.ptr, in_dim, out_dim, batch_size, m->aggr, accOutput.rect.volume(), stream);
  if (m->profiling) {
    checkCUDA(hipDeviceSynchronize());
    print_tensor<float>(accOutput.ptr, accOutput.rect.volume(), "[Embedding:backward:output_grad]");
    print_tensor<float>(accWeightGrad.ptr, accWeightGrad.rect.volume(), "[Embedding:backward:weight_grad]");
    print_tensor<int64_t>(accInput.ptr, accInput.rect.volume(), "[Embedding:backward:input]");
  }
}

__global__
void rand_generate_int64(int64_t* ptr, size_t size, int64_t p)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    ptr[i] = i % p;
  }
}

bool Embedding::measure_operator_cost(Simulator* sim,
                                      const ParallelConfig& pc,
                                      CostMetrics& cost_metrics) const
{
  ParallelTensorBase sub_input, sub_output;
  if (!outputs[0]->get_output_sub_tensor(pc, sub_output, op_type)) {
    return false;
  }
  if (!inputs[0]->get_input_sub_tensor(pc, sub_input, op_type)) {
    return false;
  }

  sim->free_all();
  bool out_of_memory = false;
  int64_t *input_ptr = (int64_t *)sim->allocate(sub_input.get_volume(), DT_INT64);
  out_of_memory = out_of_memory || (input_ptr == NULL);
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  out_of_memory = out_of_memory || (output_ptr == NULL);
  float *weight_ptr = (float *)sim->allocate(num_entries * out_channels, DT_FLOAT);
  out_of_memory = out_of_memory || (weight_ptr == NULL);
  if (out_of_memory) {
    cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    return true;
  }
  int in_dim = sub_input.dims[0].size;
  int out_dim = sub_input.dims[0].size;
  assert (sub_input.dims[1] == sub_output.dims[1]);
  int batch_size = sub_input.dims[1].size;

  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  // Randomly initialize the intput tensor to avoid out of index range issues
  hipLaunchKernelGGL(rand_generate_int64, GET_BLOCKS(sub_input.get_volume()), CUDA_NUM_THREADS, 0, stream,
      input_ptr, sub_input.get_volume(), num_entries);
  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(input_ptr, output_ptr, weight_ptr, in_dim, out_dim, batch_size, this->aggr, sub_output.get_volume(), stream);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *weight_grad_ptr = (float *)sim->allocate(num_entries * out_channels, DT_FLOAT);
    out_of_memory = out_of_memory || (weight_grad_ptr == NULL);
    float *output_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    out_of_memory = out_of_memory || (output_grad_ptr == NULL);
    int64_t *input_grad_ptr = (int64_t *)sim->allocate(sub_input.get_volume(), DT_INT64);
    out_of_memory = out_of_memory || (input_grad_ptr == NULL);
    if (out_of_memory) {
      cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
      cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
      return true;
    }
    backward = [&] {
      backward_kernel(input_grad_ptr, output_grad_ptr, weight_grad_ptr, in_dim, out_dim, batch_size,
        this->aggr, sub_output.get_volume(), stream);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure Embedding] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Measure Embedding] name(%s) forward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time);
  }

  return true;
}

}; // namespace FlexFlow

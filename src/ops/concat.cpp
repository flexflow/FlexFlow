/* Copyright 2017 Stanford, NVIDIA
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
#include "flexflow/ops/concat.h"
#include "flexflow/utils/hash_utils.h"
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
OpMeta* Concat::init_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  Concat* cc = (Concat*) task->args;
  FFHandler handler = *((const FFHandler*) task->local_args);
  ConcatMeta* m = new ConcatMeta(handler);
  // Note that our internal axis index ordering is opposite to other frameworks
  cc->init_meta(m);
  m->profiling = cc->profiling;
  return m;
}

template<int N>
void calc_blk_size(coord_t& num_blocks,
                   coord_t& blk_size,
                   Rect<N> rect,
                   int axis)
{
  num_blocks = 1;
  blk_size = 1;
  for (int d = 0; d < N; d++) {
    if (d <= axis)
      blk_size *= (rect.hi[d] - rect.lo[d] + 1);
    else
      num_blocks *= (rect.hi[d] - rect.lo[d] + 1);
  }
}

/*static*/
void Concat::forward_kernel(float* output,
                            float const * const *inputs,
                            int num_inputs,
                            int axis,
                            const Domain& out_domain,
                            const Domain* in_domain,
                            hipStream_t stream)
{
  coord_t num_blocks = 1, output_blk_size = 1, input_blk_sizes[MAX_NUM_INPUTS];
  assert(num_inputs <= MAX_NUM_INPUTS);
  switch (out_domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = out_domain; \
      calc_blk_size<DIM>(num_blocks, output_blk_size, rect, axis); \
      for (int i = 0; i < num_inputs; i++) { \
        rect = in_domain[i]; \
        coord_t input_num_blocks = 1; \
        calc_blk_size<DIM>(input_num_blocks, input_blk_sizes[i], rect, axis); \
        assert(input_num_blocks == num_blocks); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      fprintf(stderr, "Unsupported concat dimension number");
      assert(false);
  }

  for (int i = 0; i < num_inputs; i++) {
    hipLaunchKernelGGL(copy_with_stride, GET_BLOCKS(input_blk_sizes[i]*num_blocks), CUDA_NUM_THREADS, 0, stream,
        output, inputs[i], num_blocks, output_blk_size, input_blk_sizes[i]);
    //printf("output = %x num_blocks=%d output_blk_size=%d input_blk_size[%d]=%d\n",
    //       output, num_blocks, output_blk_size, i, input_blk_sizes[i]);
    output += input_blk_sizes[i];
  }
}

/*
  regions[0](O): output
  regions[1..numInputs](I): inputs
*/
void Concat::forward_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  const Concat* cc = (Concat*) task->args;
  // Note that our internal axis index ordering is opposite to other frameworks
  assert(regions.size() == cc->numInputs + 1);
  assert(task->regions.size() == cc->numInputs + 1);
  Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  //assert(out_domain.get_dim() == cc->outputs[0].num_dims);
  Domain in_domain[MAX_NUM_INPUTS];
  for (int i = 0; i < cc->numInputs; i++)
    in_domain[i] = runtime->get_index_space_domain(
        ctx, task->regions[i+1].region.get_index_space());
  float *output = helperGetTensorPointerWO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  const float *inputs[MAX_NUM_INPUTS];
  for (int i = 0; i < cc->numInputs; i++)
    inputs[i] = helperGetTensorPointerRO<float>(
        regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime);
  
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  
  hipEvent_t t_start, t_end;
  if (cc->profiling) {
    hipEventCreate(&t_start);
    hipEventCreate(&t_end);
    hipEventRecord(t_start, stream);
  }
  forward_kernel(output, inputs, cc->numInputs, cc->axis, out_domain, in_domain, stream);
  if (cc->profiling) {
    hipEventRecord(t_end, stream);
    checkCUDA(hipEventSynchronize(t_end));
    //print_tensor<4, float>(output - output_blk_size, output_rect, "[Concat:forward:output]");
    //printf("output_blk_size=%zu\n", output_blk_size);
    //print_tensor<4, float>(inputs[0], input_rect[0], "[Concat:forward:input0]");
    //print_tensor<4, float>(inputs[1], input_rect[1], "[Concat:forward:input1]");
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    printf("[%s] forward time = %.4f ms\n", cc->name, elapsed);
    hipEventDestroy(t_start);
    hipEventDestroy(t_end);
  }
}

void Concat::backward_kernel(const float* output_grad,
                             float** input_grads,
                             int num_inputs,
                             int axis,
                             const Domain& out_grad_domain,
                             const Domain* in_grad_domain,
                             hipStream_t stream)
{
  coord_t num_blocks = 1, output_blk_size = 1, input_blk_sizes[MAX_NUM_INPUTS];
  assert(num_inputs <= MAX_NUM_INPUTS);
  switch (out_grad_domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = out_grad_domain; \
      calc_blk_size<DIM>(num_blocks, output_blk_size, rect, axis); \
      for (int i = 0; i < num_inputs; i++) { \
        rect = in_grad_domain[i]; \
        coord_t input_num_blocks = 1; \
        calc_blk_size<DIM>(input_num_blocks, input_blk_sizes[i], rect, axis); \
        assert(input_num_blocks == num_blocks); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      fprintf(stderr, "Unsupported concat dimension number");
      assert(false);
  }

  for (int i = 0; i < num_inputs; i++) {
    hipLaunchKernelGGL(add_with_stride, GET_BLOCKS(input_blk_sizes[i]*num_blocks), CUDA_NUM_THREADS, 0, stream,
        input_grads[i], output_grad, num_blocks, input_blk_sizes[i], output_blk_size);
    output_grad += input_blk_sizes[i];
  }

  //Rect<2> output_rect(Point<2>(0, 0), Point<2>(output_blk_size-1, batch_size - 1));
  //Rect<2> input_rect(Point<2>(0, 0), Point<2>(input_blk_sizes[0]-1, batch_size - 1));
  //print_tensor<2, float>(output_grad - output_blk_size, output_rect, "[Concat:backward:output]");
  //print_tensor<2, float>(input_grads[0], input_rect, "[Concat:backward:input0]");
}

/*
  regions[0](I): output_grad
  regions[1..numInputs](I/O): input_grad
*/
void Concat::backward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  const Concat* cc = (Concat*) task->args;
  // Note that our internal axis index ordering is opposite to other frameworks
  assert(regions.size() == cc->numInputs + 1);
  assert(task->regions.size() == cc->numInputs + 1);
  assert(cc->numInputs <= MAX_NUM_INPUTS);
  Domain out_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  //assert(out_grad_domain.get_dim() == cc->outputs[0].num_dims);
  Domain in_grad_domains[MAX_NUM_INPUTS];
  for (int i = 0; i < cc->numInputs; i++)
    in_grad_domains[i] = runtime->get_index_space_domain(
        ctx, task->regions[i+1].region.get_index_space());
  const float *output_grad = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float *input_grads[MAX_NUM_INPUTS];
  for (int i = 0; i < cc->numInputs; i++)
    input_grads[i] = helperGetTensorPointerRW<float>(
        regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime);

  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  
  hipEvent_t t_start, t_end;
  if (cc->profiling) {
    hipEventCreate(&t_start);
    hipEventCreate(&t_end);
    hipEventRecord(t_start, stream);
  }
  backward_kernel(output_grad, input_grads, cc->numInputs, cc->axis,
      out_grad_domain, in_grad_domains, stream);
  if (cc->profiling) {
    hipEventRecord(t_end, stream);
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    printf("[%s] forward time = %.4f ms\n", cc->name, elapsed);
    hipEventDestroy(t_start);
    hipEventDestroy(t_end);
  }
}

bool Concat::measure_operator_cost(Simulator* sim,
                                   const ParallelConfig& pc,
                                   CostMetrics& cost_metrics) const
{
  assert (numInputs <= MAX_NUM_INPUTS);
  ParallelTensorBase sub_inputs[MAX_NUM_INPUTS], sub_output;
  if (!outputs[0]->get_output_sub_tensor(pc, sub_output, op_type)) {
    return false;
  }
  for (int i = 0; i < numInputs; i++) {
    if (!inputs[i]->get_input_sub_tensor(pc, sub_inputs[i], op_type)) {
      return false;
    }
  }

  ConcatMeta *m = sim->concat_meta;
  this->init_meta(m);

  sim->free_all();
  float *input_ptrs[MAX_NUM_INPUTS];
  float *input_grad_ptrs[MAX_NUM_INPUTS];
  bool out_of_memory = false;
  for (int i = 0; i < numInputs; i++) {
    input_ptrs[i] = (float *)sim->allocate(sub_inputs[i].get_volume(), DT_FLOAT);
    out_of_memory = out_of_memory || (input_ptrs[i] == NULL);
  }
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  out_of_memory = out_of_memory || (output_ptr == NULL);
  if (out_of_memory) {
    cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    return true;
  }

  Domain out_domain = sub_output.get_domain();
  Domain in_domains[MAX_NUM_INPUTS];
  for (int i = 0; i < numInputs; i++) {
    in_domains[i] = sub_inputs[i].get_domain();
  }
  
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(output_ptr, input_ptrs, numInputs, axis, out_domain, in_domains, stream);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    for (int i = 0; i < numInputs; i++) {
      input_grad_ptrs[i] = (float *)sim->allocate(sub_inputs[i].get_volume(), DT_FLOAT);
      out_of_memory = out_of_memory || (input_grad_ptrs[i] == NULL);
    }
    float *output_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    out_of_memory = out_of_memory || (output_grad_ptr == NULL);
    if (out_of_memory) {
      cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
      cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
      return true;
    }
    backward = [&] {
      backward_kernel(output_grad_ptr, input_grad_ptrs,
        numInputs, axis, out_domain, in_domains, stream);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure Concat] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Measure Concat] name(%s) forward_time(%.4lf)\n",
        name, cost_metrics.forward_time);
  }

  return true;
}

}; // namespace FlexFlow

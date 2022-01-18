/* Copyright 2020 Facebook
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
#include "flexflow/ops/reverse.h"
#include "flexflow/utils/hip_helper.h"

namespace FlexFlow {

__global__
void reverse_forward_kernel(const float* in_ptr,
                            float* out_ptr,
                            coord_t num_out_blks,
                            coord_t reverse_dim_size,
                            coord_t in_blk_size)
{
  CUDA_KERNEL_LOOP(i, num_out_blks * reverse_dim_size * in_blk_size)
  {
    coord_t blk_idx = i / (reverse_dim_size * in_blk_size);
    i = i - blk_idx * (reverse_dim_size * in_blk_size);
    coord_t reverse_dim_idx = i / in_blk_size;
    i = i - reverse_dim_idx * in_blk_size;
    coord_t in_idx = blk_idx * (reverse_dim_size * in_blk_size)
                   + (reverse_dim_size - 1 - reverse_dim_idx) * in_blk_size + i;
    out_ptr[i] = in_ptr[in_idx];
  }
}

/*static*/
void Reverse::forward_kernel(float const *in_ptr,
                             float *out_ptr,
                             coord_t num_out_blks,
                             coord_t reverse_dim_size,
                             coord_t in_blk_size,
                             coord_t output_size,
                             hipStream_t stream)
{
  hipLaunchKernelGGL(reverse_forward_kernel, GET_BLOCKS(output_size), CUDA_NUM_THREADS, 0, stream, 
      in_ptr, out_ptr, num_out_blks, reverse_dim_size, in_blk_size);
}

/*static*/
void Reverse::forward_kernel_wrapper(float const *in_ptr,
                                     float *out_ptr,
                                     coord_t num_out_blks,
                                     coord_t reverse_dim_size,
                                     coord_t in_blk_size,
                                     coord_t output_size)
{
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  Reverse::forward_kernel(in_ptr, out_ptr, num_out_blks, reverse_dim_size, in_blk_size, output_size, stream);
}

void Reverse::backward_kernel(float const *out_grad_ptr,
                              float *in_grad_ptr,
                              coord_t num_out_blks,
                              coord_t reverse_dim_size,
                              coord_t in_blk_size,
                              coord_t input_size,
                              hipStream_t stream)
{
  hipLaunchKernelGGL(reverse_forward_kernel, GET_BLOCKS(input_size), CUDA_NUM_THREADS, 0, stream, 
      out_grad_ptr, in_grad_ptr, num_out_blks, reverse_dim_size, in_blk_size);
}

__host__
void Reverse::backward_task(const Task* task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const Reverse* reverse = (const Reverse*) task->args;
  Domain out_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain in_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  assert(out_grad_domain == in_grad_domain);
  const float* out_grad_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* in_grad_ptr = helperGetTensorPointerRW<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  // We reuse the forward kernel for backward tasks
  int axis = in_grad_domain.get_dim() - reverse->axis - 1;
  coord_t in_blk_size = 1, reverse_dim_size = 1, num_out_blks = 1;
  for (int i = 0; i < in_grad_domain.get_dim(); i++) {
    if (i < axis)
      in_blk_size *= in_grad_domain.hi()[i] - in_grad_domain.lo()[i] + 1;
    else if (i == axis)
      reverse_dim_size = in_grad_domain.hi()[i] - in_grad_domain.lo()[i] + 1;
    else
      num_out_blks *= in_grad_domain.hi()[i] - in_grad_domain.lo()[i] + 1;
  }

  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  backward_kernel(out_grad_ptr, in_grad_ptr, num_out_blks, reverse_dim_size, in_blk_size, in_grad_domain.get_volume(), stream);
}

bool Reverse::measure_operator_cost(Simulator* sim,
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
  float *input_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert (input_ptr != NULL);
  float *output_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_ptr != NULL);

  coord_t in_blk_size = 1, reverse_dim_size = 1, num_out_blks = 1;
  for (int i = 0; i < sub_output.num_dims; i++) {
    if (i < axis) {
      in_blk_size *= sub_output.dims[i].size;
    } else if (i == axis) {
      reverse_dim_size = sub_output.dims[i].size;
    } else {
      num_out_blks *= sub_output.dims[i].size;
    }
  }

  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  std::function<void()> forward, backward;
  forward = [&] {
     forward_kernel(input_ptr, output_ptr, num_out_blks, reverse_dim_size, in_blk_size, sub_output.get_volume(), stream);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *input_grad_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    assert (input_grad_ptr != NULL);
    float *output_grad_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert (output_grad_ptr != NULL);
    backward = [&] {
      backward_kernel(output_grad_ptr, input_grad_ptr, num_out_blks, reverse_dim_size, in_blk_size, sub_input.get_volume(), stream);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure Reverse] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Measure Reverse] name(%s) forward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time);
  }

  return true;
}

}; // namespace FlexFlow

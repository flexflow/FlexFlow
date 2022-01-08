/* Copyright 2018 Stanford
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
#include "flexflow/ops/flat.h"
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

OpMeta* Flat::init_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  FFHandler handler = *((const FFHandler*) task->local_args);
  FlatMeta* m = new FlatMeta(handler);
  return m;
}

/*static*/
void Flat::forward_kernel(const float* input_ptr,
                          float* output_ptr,
                          size_t num_elements,
                          hipStream_t stream)
{
  checkCUDA(hipMemcpyAsync(output_ptr, input_ptr,
                            num_elements * sizeof(float),
                            hipMemcpyDeviceToDevice, stream));
}

/*
  regions[0](I): input
  regions[1](O): output
*/
void Flat::forward_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  TensorAccessorR<float, Input::NUMDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, Output::NUMDIM> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  assert(acc_input.rect.volume() == acc_output.rect.volume());

  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  forward_kernel(acc_input.ptr, acc_output.ptr, acc_input.rect.volume(), stream);
  //checkCUDA(hipDeviceSynchronize());
}

void Flat::backward_kernel(float* input_grad_ptr,
                           const float* output_grad_ptr,
                           size_t num_elements,
                           hipStream_t stream)
{
  float alpha = 1.0f;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(apply_add_with_scale<float>), GET_BLOCKS(num_elements), CUDA_NUM_THREADS, 0, stream, 
      input_grad_ptr, output_grad_ptr, num_elements, alpha);
}

/*
  regions[0](I/O) : input_grad
  regions[1](I) : output_grad
*/
void Flat::backward_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  TensorAccessorW<float, Input::NUMDIM> acc_input_grad(
    regions[0], task->regions[0], FID_DATA, ctx, runtime,
    true/*readOutput*/);
  TensorAccessorR<float, Output::NUMDIM> acc_output_grad(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  assert(acc_input_grad.rect.volume() == acc_output_grad.rect.volume());

  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  backward_kernel(acc_input_grad.ptr, acc_output_grad.ptr, acc_input_grad.rect.volume(), stream);
  //checkCUDA(hipMemcpyAsync(acc_input_grad.ptr, acc_output_grad.ptr,
  //                          acc_input_grad.rect.volume() * sizeof(float),
  //                          hipMemcpyDeviceToDevice));
  //checkCUDA(hipDeviceSynchronize());
}

bool Flat::measure_operator_cost(Simulator* sim,
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
  float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert (input_ptr != NULL);
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_ptr != NULL);
  size_t num_elements = sub_output.get_volume();

  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(input_ptr, output_ptr, num_elements, stream);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *input_grad_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    float *output_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert (output_grad_ptr != NULL);
    assert (input_grad_ptr != NULL);
    backward = [&] {
      backward_kernel(input_grad_ptr, output_grad_ptr, num_elements, stream);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    log_measure.debug("[Measure Flat] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    log_measure.debug("[Measure Flat] name(%s) forward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time);
  }

  return true;
}

}; // namespace FlexFlow

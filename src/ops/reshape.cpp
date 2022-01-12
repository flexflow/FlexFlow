/* Copyright 2020 Stanford, Facebook
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
#include "flexflow/ops/reshape.h"
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

ReshapeMeta::ReshapeMeta(FFHandler handler)
: OpMeta(handler) {}

OpMeta* Reshape::init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  const Reshape* reshape = (Reshape*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  ReshapeMeta* m = new ReshapeMeta(handle);
  m->data_type = reshape->outputs[0]->data_type;
  return m;
}

/*static*/
template<typename T>
void Reshape::forward_kernel(const T* input_ptr,
                             T* output_ptr,
                             size_t num_elements,
                             hipStream_t stream)
{
  checkCUDA(hipMemcpyAsync(output_ptr, input_ptr,
      num_elements * sizeof(T), hipMemcpyDeviceToDevice, stream));
}

void Reshape::forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Reshape* reshape = (const Reshape*) task->args;
  const ReshapeMeta* m = *((ReshapeMeta**) task->local_args);
  Domain in_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain out_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  assert(in_domain.get_volume() == out_domain.get_volume());
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  if (m->data_type == DT_FLOAT) {
    const float* in_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
    float* out_ptr = helperGetTensorPointerWO<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
    forward_kernel<float>(in_ptr, out_ptr, in_domain.get_volume(), stream);
  } else if (m->data_type == DT_DOUBLE) {
    const double* in_ptr = helperGetTensorPointerRO<double>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
    double* out_ptr = helperGetTensorPointerWO<double>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
    forward_kernel<double>(in_ptr, out_ptr, in_domain.get_volume(), stream);
  } else if (m->data_type == DT_INT32) {
    const int32_t* in_ptr = helperGetTensorPointerRO<int32_t>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
    int32_t* out_ptr = helperGetTensorPointerWO<int32_t>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
    forward_kernel<int32_t>(in_ptr, out_ptr, in_domain.get_volume(), stream);
  } else if (m->data_type == DT_INT64) {
    const int64_t* in_ptr = helperGetTensorPointerRO<int64_t>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
    int64_t* out_ptr = helperGetTensorPointerWO<int64_t>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
    forward_kernel<int64_t>(in_ptr, out_ptr, in_domain.get_volume(), stream);
  } else {
    assert(false && "Unsupported data type in Reshape forward");
  }
}

template<typename T>
void Reshape::backward_kernel(T* input_grad_ptr,
                              const T* output_grad_ptr,
                              size_t num_elements,
                              hipStream_t stream)
{
  float alpha = 1.0f;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(apply_add_with_scale<T>), GET_BLOCKS(num_elements), CUDA_NUM_THREADS, 0, stream, 
      input_grad_ptr, output_grad_ptr, num_elements, (T)alpha);
}

void Reshape::backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Reshape* reshape = (const Reshape*) task->args;
  const ReshapeMeta* m = *((ReshapeMeta**) task->local_args);
  Domain out_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain in_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  assert(in_grad_domain.get_volume() == out_grad_domain.get_volume());
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  if (m->data_type == DT_FLOAT) {
    const float* out_grad_ptr = helperGetTensorPointerRO<float>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    float* in_grad_ptr = helperGetTensorPointerRW<float>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    backward_kernel<float>(in_grad_ptr, out_grad_ptr, in_grad_domain.get_volume(), stream);
  } else if (m->data_type == DT_DOUBLE) {
    const double* out_grad_ptr = helperGetTensorPointerRO<double>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    double* in_grad_ptr = helperGetTensorPointerRW<double>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    backward_kernel<double>(in_grad_ptr, out_grad_ptr, in_grad_domain.get_volume(), stream);
  } else if (m->data_type == DT_INT32) {
    const int32_t* out_grad_ptr = helperGetTensorPointerRO<int32_t>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    int32_t* in_grad_ptr = helperGetTensorPointerRW<int32_t>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    backward_kernel<int32_t>(in_grad_ptr, out_grad_ptr, in_grad_domain.get_volume(), stream);
  } else if (m->data_type == DT_INT64) {
    const int64_t* out_grad_ptr = helperGetTensorPointerRO<int64_t>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    int64_t* in_grad_ptr = helperGetTensorPointerRW<int64_t>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    backward_kernel<int64_t>(in_grad_ptr, out_grad_ptr, in_grad_domain.get_volume(), stream);
  } else {
    assert(false && "Unsupported data type in Reshape backward");
  }
}

bool Reshape::measure_operator_cost(Simulator* sim,
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
  assert(input_ptr != NULL);
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_ptr != NULL);
  assert (sub_output.get_volume() == sub_input.get_volume());
  size_t num_elements = sub_input.get_volume();

  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(input_ptr, output_ptr, num_elements, stream);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *input_grad_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    assert(input_grad_ptr != NULL);
    float *output_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert (output_grad_ptr != NULL);

    backward = [&] {
      backward_kernel(input_grad_ptr, output_grad_ptr, num_elements, stream);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Meausre Reshape] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Meausre Reshape] name(%s) forward_time(%.4lf)\n",
        name, cost_metrics.forward_time);
  }
  return true;
}

template void Reshape::forward_kernel<float>(const float* input_ptr, float* output_ptr, size_t volume, cudaStream_t stream);
template void Reshape::backward_kernel<float>(float* in_grad_ptr, const float* out_grad_ptr, size_t volume, cudaStream_t stream);

}; // namespace FlexFlow

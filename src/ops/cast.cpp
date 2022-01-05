/* Copyright 2021 CMU
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
#include "flexflow/ops/cast.h"
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


OpMeta* Cast::init_task(
    const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime) {
  Cast* cast = (Cast*) task->args;
  FFHandler handler = *((const FFHandler*) task->local_args);
  CastMeta* m = new CastMeta(handler);
  m->input_data_type = cast->inputs[0]->data_type;
  m->output_data_type = cast->outputs[0]->data_type;
  return m;
}

template<typename IDT, typename ODT>
__global__
void cast_forward(
    const IDT* input,
    ODT* output,
    size_t volume) {
  CUDA_KERNEL_LOOP(i, volume) {
    output[i] = (ODT) input[i];
  }
}

template<typename IDT, typename ODT>
void Cast::forward_kernel(
    const IDT* input_ptr,
    ODT* output_ptr,
    size_t volume,
    hipStream_t stream) {
  hipLaunchKernelGGL(HIP_KERNEL_NAME(cast_forward<IDT, ODT>), GET_BLOCKS(volume), CUDA_NUM_THREADS, 0, stream, 
      input_ptr, output_ptr, volume);
}

template<typename IDT>
void Cast::forward_task_with_1_type(
    const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime) {
  const CastMeta* m = *((CastMeta**) task->local_args);
  if (m->output_data_type == DT_FLOAT) {
    Cast::forward_task_with_2_type<IDT, float>(task, regions, ctx, runtime);
  } else if (m->output_data_type == DT_DOUBLE) {
    Cast::forward_task_with_2_type<IDT, double>(task, regions, ctx, runtime);
  } else if (m->output_data_type == DT_INT32) {
    Cast::forward_task_with_2_type<IDT, int32_t>(task, regions, ctx, runtime);
  } else if (m->output_data_type == DT_INT64) {
    Cast::forward_task_with_2_type<IDT, int64_t>(task, regions, ctx, runtime);
  }
}

template<typename IDT, typename ODT>
void Cast::forward_task_with_2_type(
    const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == regions.size());
  Domain input_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  const IDT* input_ptr = helperGetTensorPointerRO<IDT>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  ODT* output_ptr = helperGetTensorPointerWO<ODT>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  forward_kernel<IDT, ODT>(input_ptr, output_ptr, output_domain.get_volume(), stream);
}

void Cast::forward_task(
    const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime) {
  const CastMeta* m = *((CastMeta**) task->local_args);
  if (m->input_data_type == DT_FLOAT) {
    Cast::forward_task_with_1_type<float>(task, regions, ctx, runtime);
  } else if (m->input_data_type == DT_DOUBLE) {
    Cast::forward_task_with_1_type<double>(task, regions, ctx, runtime);
  } else if (m->input_data_type == DT_INT32) {
    Cast::forward_task_with_1_type<int32_t>(task, regions, ctx, runtime);
  } else if (m->input_data_type == DT_INT64) {
    Cast::forward_task_with_1_type<int64_t>(task, regions, ctx, runtime);
  }
}

template<typename IDT, typename ODT>
__global__
void cast_backward(
    const IDT* input,
    ODT* output,
    size_t volume,
    ODT beta) {
  CUDA_KERNEL_LOOP(i, volume) {
    output[i] = (ODT) input[i] + beta * output[i];
  }
}

template<typename IDT, typename ODT>
void Cast::backward_kernel(
    const IDT* src_ptr,
    ODT* dst_ptr,
    size_t volume,
    hipStream_t stream) {
  hipLaunchKernelGGL(HIP_KERNEL_NAME(cast_backward<IDT, ODT>), GET_BLOCKS(volume), CUDA_NUM_THREADS, 0, stream, 
      src_ptr, dst_ptr, volume, (ODT)1.0f);
}

template<typename IDT>
void Cast::backward_task_with_1_type(
    const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime) {
  const CastMeta* m = *((CastMeta**) task->local_args);
  if (m->input_data_type == DT_FLOAT) {
    Cast::backward_task_with_2_type<IDT, float>(task, regions, ctx, runtime);
  } else if (m->input_data_type == DT_DOUBLE) {
    Cast::backward_task_with_2_type<IDT, double>(task, regions, ctx, runtime);
  } else if (m->input_data_type == DT_INT32) {
    Cast::backward_task_with_2_type<IDT, int32_t>(task, regions, ctx, runtime);
  } else if (m->input_data_type == DT_INT64) {
    Cast::backward_task_with_2_type<IDT, int64_t>(task, regions, ctx, runtime);
  }
}

template<typename IDT, typename ODT>
void Cast::backward_task_with_2_type(
    const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == regions.size());
  Domain input_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  const IDT* input_ptr = helperGetTensorPointerRO<IDT>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  ODT* output_ptr = helperGetTensorPointerRW<ODT>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  backward_kernel<IDT, ODT>(input_ptr, output_ptr, output_domain.get_volume(), stream);
}

void Cast::backward_task(
    const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime) {
  const CastMeta* m = *((CastMeta**) task->local_args);
  if (m->output_data_type == DT_FLOAT) {
    Cast::backward_task_with_1_type<float>(task, regions, ctx, runtime);
  } else if (m->output_data_type == DT_DOUBLE) {
    Cast::backward_task_with_1_type<double>(task, regions, ctx, runtime);
  } else if (m->output_data_type == DT_INT32) {
    Cast::backward_task_with_1_type<int32_t>(task, regions, ctx, runtime);
  } else if (m->output_data_type == DT_INT64) {
    Cast::backward_task_with_1_type<int64_t>(task, regions, ctx, runtime);
  }
}

bool Cast::measure_operator_cost(
    Simulator*sim,
    const ParallelConfig& pc,
    CostMetrics& cost_metrics) const {
  // Assume cast has no cost
  cost_metrics.forward_time = 0.0f;
  cost_metrics.backward_time = 0.0f;
  return true;
}

CastMeta::CastMeta(FFHandler handle)
: OpMeta(handle) {}

}; //namespace FlexFlow

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

#include "model.h"
#include "cuda_helper.h"

Tensor FFModel::cast(const Tensor& input,
                     DataType dtype,
                     const char* name)
{
  Cast* cast = new Cast(*this, input, dtype, name);
  layers.push_back(cast);
  return cast->outputs[0];
}

Cast::Cast(FFModel& model,
           const Tensor& input,
           DataType _dtype,
           const char* name)
: Op(model, OP_CAST, name, input)
{
  numOutputs = 1;
  numWeights = 0;
  outputs[0].numDim = input.numDim;
  outputs[0].data_type = _dtype;
  for (int i = 0; i < outputs[0].numDim; i++)
    outputs[0].adim[i] = input.adim[i];
}

void Cast::create_weights(FFModel& model)
{}

void Cast::create_output_and_partition(FFModel& model)
{
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = model.get_or_create_task_is(inputs[0].numDim, pcname);

  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  int dims[MAX_TENSOR_DIM];
  for (int i = 0; i < outputs[0].numDim; i++)
    dims[i] = outputs[0].adim[outputs[0].numDim-1-i];
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> part_rect = domain; \
      outputs[0] = model.create_tensor<DIM>(dims, outputs[0].data_type, this); \
      outputs[0].owner_op = this; \
      outputs[0].owner_idx = 0; \
      Rect<DIM> input_rect = runtime->get_index_partition_color_space( \
          ctx, inputs[0].part.get_index_partition()); \
      if (input_rect == part_rect) { \
        input_lps[0] = inputs[0].part; \
        input_grad_lps[0] = inputs[0].part_grad; \
      } else { \
        model.create_disjoint_partition<DIM>(inputs[0], \
            IndexSpaceT<DIM>(task_is), input_lps[0], input_grad_lps[0]); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC);
#undef DIMFUNC
    default:
    {
      fprintf(stderr, "Unsupported concat dimension number");
      assert(false);
    }
  }
}

OpMeta* Cast::init_task(
    const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime) {
  Cast* cast = (Cast*) task->args;
  FFHandler handler = *((const FFHandler*) task->local_args);
  CastMeta* m = new CastMeta(handler);
  m->input_data_type = cast->inputs[0].data_type;
  m->output_data_type = cast->outputs[0].data_type;
  return m;
}

void Cast::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      ParallelConfig pc; \
      std::string pcname = name; \
      ff.config.find_parallel_config(DIM, pcname, pc); \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        FFHandler handle = ff.handlers[pc.device_ids[idx++]]; \
        argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler))); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  IndexLauncher launcher(CAST_INIT_TASK_ID, task_is,
    TaskArgument(this, sizeof(Cast)), argmap,
    Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
    FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        meta[idx++] = fm.get_result<OpMeta*>(*it); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
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
    cudaStream_t stream) {
  cast_forward<IDT, ODT><<<GET_BLOCKS(volume), CUDA_NUM_THREADS, 0, stream>>>(
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
  cudaStream_t stream;
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

void Cast::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        OpMeta* mp = meta[idx++]; \
        argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*))); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  IndexLauncher launcher(CAST_FWD_TASK_ID, task_is,
                         TaskArgument(NULL, false), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
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
    cudaStream_t stream) {
  cast_backward<IDT, ODT><<<GET_BLOCKS(volume), CUDA_NUM_THREADS, 0, stream>>>(
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
  cudaStream_t stream;
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

void Cast::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        OpMeta* mp = meta[idx++]; \
        argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*))); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  IndexLauncher launcher(CAST_BWD_TASK_ID, task_is,
                         TaskArgument(NULL, false), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, outputs[0].region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

bool Cast::measure_operator_cost(
    Simulator*sim,
    const ParallelConfig& pc,
    CostMetrics& cost_metrics) {
  // Assume cast has no cost
  cost_metrics.forward_time = 0.0f;
  cost_metrics.backward_time = 0.0f;
  return true;
}

CastMeta::CastMeta(FFHandler handle)
: OpMeta(handle) {}

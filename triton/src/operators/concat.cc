/* Copyright 2022 NVIDIA CORPORATION
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

#include "concat.h"

using namespace Legion;

namespace triton { namespace backend { namespace legion {

Legion::ProjectionID Concat::filter_functor_id;

LogicalRegion
FilterProjectionFunctor::project(
    LogicalPartition upper_bound, const DomainPoint& point,
    const Domain& domain)
{
  // Check to see if the point is in the color space
  const Domain limits = runtime->get_index_partition_color_space(
      upper_bound.get_index_partition());
  if (limits.contains(point))
    return runtime->get_logical_subregion_by_color(upper_bound, point);
  else
    return LogicalRegion::NO_REGION;
}

ConcatArgs::ConcatArgs(void) : local_index(0), datatype(DT_NONE), axis(-1) {}

Concat::Concat(
    LegionModelState* model, const LayerStrategy* strategy, size_t inputs,
    int ax, const char* name)
    : Operator(model, strategy, OperatorType::OP_CONCAT, name, inputs, 0, 1),
      axis(ax)
{
  assert(inputs > 0);
}

void
Concat::Configure(const std::vector<Tensor*>& ins, Tensor* out)
{
  assert(num_inputs == ins.size());
  inputs = ins;
  size_t axis_size = 0;
  const size_t dims = out->bounds.size();
  assert(dims == strategy->nDims);
  for (unsigned idx = 0; idx < inputs.size(); idx++) {
    assert(inputs[idx]->type == out->type);
    assert(inputs[idx]->bounds.size() == dims);
    for (unsigned d = 0; d < dims; d++) {
      if (d == axis)
        axis_size += inputs[idx]->bounds[d];
      else
        assert(inputs[idx]->bounds[d] == out->bounds[d]);
    }
  }
  assert(axis_size == out->bounds[axis]);
  outputs.push_back(out);
  // Figure out the output tiling domain
  std::vector<size_t> tile_sizes(dims);
  for (unsigned d = 0; d < dims; d++)
    tile_sizes[d] = (out->bounds[d] + strategy->dim[d] - 1) / strategy->dim[d];
  coord_t offset = 0;
  // Now compute the domains and transforms needed for constructing
  // the partitions for each of the inputs
  input_color_spaces.resize(num_inputs);
  input_extents.resize(num_inputs);
  for (unsigned idx = 0; idx < num_inputs; idx++) {
    DomainPoint lo, hi, color_lo, color_hi;
    lo.dim = dims;
    hi.dim = dims;
    color_lo.dim = dims;
    color_hi.dim = dims;
    for (int d = 0; d < dims; d++) {
      if (d == axis) {
        const coord_t extent = inputs[idx]->bounds[d];
        lo[d] = -offset;
        hi[d] = (tile_sizes[d] - 1 /*inclusive*/) - offset;
        color_lo[d] = offset / tile_sizes[d];
        color_hi[d] = (offset + extent - 1) / tile_sizes[d];
        offset += extent;
      } else {
        lo[d] = 0;
        hi[d] = tile_sizes[d] - 1;  // make it inclusive
        color_lo[d] = 0;
        color_hi[d] = strategy->dim[d] - 1;  // make it inclusive
      }
    }
    input_color_spaces[idx] = Domain(color_lo, color_hi);
    input_extents[idx] = Domain(lo, hi);
  }
  // The input transform is the same across all the inputs
  switch (dims) {
#define DIMFUNC(N)                         \
  case N: {                                \
    Transform<N, N> transform;             \
    for (int i = 0; i < N; i++)            \
      for (int j = 0; j < N; j++)          \
        if (i == j)                        \
          transform[i][j] = tile_sizes[i]; \
        else                               \
          transform[i][j] = 0;             \
    input_transform = transform;           \
    break;                                 \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      abort();
  }
}

Domain
Concat::GetBounds(Processor proc)
{
  const size_t dims = outputs[0]->bounds.size();
  DomainPoint lo, hi;
  lo.dim = dims;
  hi.dim = dims;
  for (int d = 0; d < dims; d++) {
    lo[d] = 0;
    hi[d] = outputs[0]->bounds[d] - 1;
  }
  const Domain global(lo, hi);
  return strategy->find_local_domain(proc, global);
}

void
Concat::Load(Processor proc)
{
  assert(proc.kind() == strategy->kind);
  assert(inputs[0]->bounds.size() == size_t(strategy->nDims));
  // If this processor is not used for this layer there is nothing to do
  if (!strategy->is_local_processor(proc))
    return;
  const unsigned local_index = strategy->find_local_offset(proc);
  ConcatArgs& proc_args = args[local_index];
  proc_args.owner = this;
  proc_args.local_index = local_index;
  proc_args.bounds = GetBounds(proc);
  proc_args.datatype = inputs[0]->type;
  proc_args.axis = axis;
}

void
Concat::initialize(
    LegionModelInstance* instance, const unsigned instance_index,
    Runtime* runtime, Context ctx, MapperID mapper)
{
  const Domain launch_domain = strategy->get_launch_domain();
  // Find or create the launch space domain
  IndexSpace launch_space = instance->find_or_create_index_space(launch_domain);
  // Also get the sharding function from the strategy
  ShardingFunction* shardfn = strategy->sharding_function;
  // Construct a future map for the pass-by-value arguments
  std::map<DomainPoint, TaskArgument> values;
  for (Domain::DomainPointIterator itr(launch_domain); itr; itr++) {
    const Processor proc = shardfn->find_proc(itr.p, launch_domain);
    if (!strategy->is_local_processor(proc))
      continue;
    const unsigned local_index = strategy->find_local_offset(proc);
    values[itr.p] = TaskArgument(args + local_index, sizeof(ConcatArgs));
  }
  argmaps[instance_index] = runtime->construct_future_map(
      ctx, launch_space, values, true /*collective*/, shardfn->sharding_id);

  IndexTaskLauncher& launcher = launchers[instance_index];
  launcher = IndexTaskLauncher(
      CONCAT_TASK_ID, launch_space, TaskArgument(NULL, 0),
      ArgumentMap(argmaps[instance_index]), Predicate::TRUE_PRED,
      false /*must*/, mapper, strategy->tag);
  LogicalRegion output_region = instance->create_tensor_region(outputs[0]);
  LogicalPartition output_part =
      instance->find_or_create_tiled_partition(outputs[0], strategy);
  launcher.add_region_requirement(RegionRequirement(
      output_part, 0 /*projection id*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE,
      output_region));
  launcher.add_field(0, FID_DATA);
  assert(inputs.size() == input_color_spaces.size());
  assert(inputs.size() == input_extents.size());
  // Now go through and create partitions for each of the input regions
  for (unsigned idx = 0; idx < inputs.size(); idx++) {
    IndexSpace input_color_space =
        instance->find_or_create_index_space(input_color_spaces[idx]);
    LogicalRegion input_region = inputs[idx]->region[instance_index];
    IndexPartition index_part = instance->find_or_create_partition(
        input_region.get_index_space(), input_color_space, input_transform,
        input_extents[idx], LEGION_DISJOINT_COMPLETE_KIND);
    LogicalPartition input_part = runtime->get_logical_partition_by_tree(
        ctx, index_part, input_region.get_field_space(),
        input_region.get_tree_id());
    launcher.add_region_requirement(RegionRequirement(
        input_part, filter_functor_id, LEGION_READ_ONLY, LEGION_EXCLUSIVE,
        input_region));
    launcher.add_field(idx + 1 /*include output*/, FID_DATA);
  }
}

void
Concat::forward(
    LegionModelInstance* instance, const unsigned instance_index,
    Runtime* runtime, Context ctx, MapperID mapper)
{
  runtime->execute_index_space(ctx, launchers[instance_index]);
}

void
Concat::finalize(
    LegionModelInstance* instance, const unsigned instance_index,
    Runtime* runtime, Context ctx, MapperID mapper)
{
  argmaps[instance_index] = FutureMap();
}

void
Concat::Free(Processor proc)
{
  assert(proc.kind() == strategy->kind);
}

/*static*/ void
Concat::PreregisterTaskVariants(void)
{
  {
    // Register our special projection functor with the runtime
    filter_functor_id = Runtime::generate_static_projection_id();
    Runtime::preregister_projection_functor(
        filter_functor_id, new FilterProjectionFunctor);
  }
  {
    TaskVariantRegistrar cpu_registrar(CONCAT_TASK_ID, "Concat CPU");
    cpu_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    cpu_registrar.set_leaf();
    Runtime::preregister_task_variant<forward_cpu>(
        cpu_registrar, "Concat Operator");
  }
#ifdef LEGION_USE_CUDA
  {
    TaskVariantRegistrar gpu_registrar(CONCAT_TASK_ID, "Concat GPU");
    gpu_registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    gpu_registrar.set_leaf();
    Runtime::preregister_task_variant<forward_gpu>(
        gpu_registrar, "Concat Operator");
  }
#endif
}

/*static*/ void
Concat::forward_cpu(
    const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
    Runtime* runtime)
{
  assert(task->local_arglen == sizeof(ConcatArgs));
  const ConcatArgs* args = (const ConcatArgs*)task->local_args;
  assert(regions.size() >= 2);
  assert(regions.size() == task->regions.size());
  uint8_t* output_ptr = nullptr;
  size_t total_elements = 1;
  size_t element_stride = sizeof_datatype(args->datatype);
  switch (args->bounds.get_dim()) {
#define DIMFUNC(DIM)                                                          \
  case DIM: {                                                                 \
    const Rect<DIM> bounds = args->bounds;                                    \
    output_ptr = (uint8_t*)TensorAccessor<LEGION_WRITE_DISCARD, DIM>::access( \
        args->datatype, bounds, regions[0]);                                  \
    for (int d = DIM - 1; d >= 0; d--) {                                      \
      element_stride *= ((bounds.hi[d] - bounds.lo[d]) + 1);                  \
      if (d == args->axis)                                                    \
        break;                                                                \
    }                                                                         \
    for (int d = 0; d < args->axis; d++)                                      \
      total_elements *= ((bounds.hi[d] - bounds.lo[d]) + 1);                  \
    break;                                                                    \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      abort();
  }
  for (unsigned idx = 1; idx < regions.size(); idx++) {
    // Skip any regions which have been masked off for this point task
    LogicalRegion region = task->regions[idx].region;
    if (!region.exists())
      continue;
    const Domain input_domain =
        runtime->get_index_space_domain(region.get_index_space());
    assert(input_domain.get_dim() == args->bounds.get_dim());
    const uint8_t* input_ptr = nullptr;
    size_t element_size = sizeof_datatype(args->datatype);
    switch (input_domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    const Rect<DIM> bounds = input_domain;                                     \
    assert(!bounds.empty());                                                   \
    input_ptr = (const uint8_t*)TensorAccessor<LEGION_READ_ONLY, DIM>::access( \
        args->datatype, bounds, regions[idx]);                                 \
    for (int d = DIM - 1; d >= 0; d--) {                                       \
      element_size *= ((bounds.hi[d] - bounds.lo[d]) + 1);                     \
      if (d == args->axis)                                                     \
        break;                                                                 \
    }                                                                          \
    for (int d = 0; d < args->axis; d++) {                                     \
      assert(bounds.lo[d] == args->bounds.lo()[d]);                            \
      assert(bounds.hi[d] == args->bounds.hi()[d]);                            \
    }                                                                          \
    break;                                                                     \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        abort();
    }
    uint8_t* current_ptr = output_ptr;
    for (size_t element = 0; element < total_elements; element++) {
      memcpy(current_ptr, input_ptr, element_size);
      input_ptr += element_size;
      current_ptr += element_stride;
    }
    // Update the output ptr with the new offset for the next set of elements
    output_ptr += element_size;
  }
}

#ifdef LEGION_USE_CUDA
/*static*/ void
Concat::forward_gpu(
    const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
    Runtime* runtime)
{
  assert(task->local_arglen == sizeof(ConcatArgs));
  const ConcatArgs* args = (const ConcatArgs*)task->local_args;
#ifndef DISABLE_LEGION_CUDA_HIJACK
  ::cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
#endif
  ::cudaEvent_t t_start, t_end;
  if (args->profiling) {
    CHECK_CUDA(cudaEventCreate(&t_start));
    CHECK_CUDA(cudaEventCreate(&t_end));
#ifdef DISABLE_LEGION_CUDA_HIJACK
    CHECK_CUDA(cudaEventRecord(t_start));
#else
    CHECK_CUDA(cudaEventRecord(t_start, stream));
#endif
  }
  assert(regions.size() >= 2);
  assert(regions.size() == task->regions.size());
  uint8_t* output_ptr = nullptr;
  size_t total_elements = 1;
  size_t element_stride = sizeof_datatype(args->datatype);
  switch (args->bounds.get_dim()) {
#define DIMFUNC(DIM)                                                          \
  case DIM: {                                                                 \
    const Rect<DIM> bounds = args->bounds;                                    \
    output_ptr = (uint8_t*)TensorAccessor<LEGION_WRITE_DISCARD, DIM>::access( \
        args->datatype, bounds, regions[0]);                                  \
    for (int d = DIM - 1; d >= 0; d--) {                                      \
      element_stride *= ((bounds.hi[d] - bounds.lo[d]) + 1);                  \
      if (d == args->axis)                                                    \
        break;                                                                \
    }                                                                         \
    for (int d = 0; d < args->axis; d++)                                      \
      total_elements *= ((bounds.hi[d] - bounds.lo[d]) + 1);                  \
    break;                                                                    \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      abort();
  }
  for (unsigned idx = 1; idx < regions.size(); idx++) {
    // Skip any regions which have been masked off for this point task
    LogicalRegion region = task->regions[idx].region;
    if (!region.exists())
      continue;
    const Domain input_domain =
        runtime->get_index_space_domain(region.get_index_space());
    assert(input_domain.get_dim() == args->bounds.get_dim());
    const uint8_t* input_ptr = nullptr;
    size_t element_size = sizeof_datatype(args->datatype);
    switch (input_domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    const Rect<DIM> bounds = input_domain;                                     \
    assert(!bounds.empty());                                                   \
    input_ptr = (const uint8_t*)TensorAccessor<LEGION_READ_ONLY, DIM>::access( \
        args->datatype, bounds, regions[idx]);                                 \
    for (int d = DIM - 1; d >= 0; d--) {                                       \
      element_size *= ((bounds.hi[d] - bounds.lo[d]) + 1);                     \
      if (d == args->axis)                                                     \
        break;                                                                 \
    }                                                                          \
    for (int d = 0; d < args->axis; d++) {                                     \
      assert(bounds.lo[d] == args->bounds.lo()[d]);                            \
      assert(bounds.hi[d] == args->bounds.hi()[d]);                            \
    }                                                                          \
    break;                                                                     \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        abort();
    }
    if (total_elements == 1) {
      assert(element_stride == element_size);
#ifdef DISABLE_LEGION_CUDA_HIJACK
      CHECK_CUDA(cudaMemcpyAsync(
          output_ptr, input_ptr, element_size, cudaMemcpyDeviceToDevice));
#else
      CHECK_CUDA(cudaMemcpyAsync(
          output_ptr, input_ptr, element_size, cudaMemcpyDeviceToDevice,
          stream));
#endif
    } else {
#ifdef DISABLE_LEGION_CUDA_HIJACK
      CHECK_CUDA(cudaMemcpy2DAsync(
          output_ptr, element_stride, input_ptr, element_size, element_size,
          total_elements, cudaMemcpyDeviceToDevice));
#else
      CHECK_CUDA(cudaMemcpy2DAsync(
          output_ptr, element_stride, input_ptr, element_size, element_size,
          total_elements, cudaMemcpyDeviceToDevice, stream));
#endif
    }
    // Update the output ptr with the new offset for the next set of elements
    output_ptr += element_size;
  }
  if (args->profiling) {
#ifdef DISABLE_LEGION_CUDA_HIJACK
    CHECK_CUDA(cudaEventRecord(t_end));
#else
    CHECK_CUDA(cudaEventRecord(t_start, stream));
#endif
    CHECK_CUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    CHECK_CUDA(cudaEventDestroy(t_start));
    CHECK_CUDA(cudaEventDestroy(t_end));
    printf(
        "%s [Concat] forward time (CF) = %.2fms\n",
        args->owner->op_name.c_str(), elapsed);
  }
}
#endif

}}}  // namespace triton::backend::legion

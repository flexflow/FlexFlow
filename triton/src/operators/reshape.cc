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

#include "reshape.h"

using namespace Legion;

namespace triton { namespace backend { namespace legion {

Reshape::Reshape(
    LegionModelState* model, const LayerStrategy* strategy, const char* name)
    : Operator(model, strategy, OperatorType::OP_RESHAPE, name, 1, 0, 1)
{
}

void
Reshape::Configure(Tensor* input, Tensor* output)
{
  assert(input != nullptr);
  assert(output != nullptr);
  assert(input->type == output->type);
  // Make sure that they have the same volumes
  size_t input_volume = 1, output_volume = 1;
  for (unsigned idx = 0; idx < input->bounds.size(); idx++)
    input_volume *= input->bounds[idx];
  for (unsigned idx = 0; idx < output->bounds.size(); idx++)
    output_volume *= output->bounds[idx];
  assert(input_volume == output_volume);

  // Group dimensions from the two input tensors together from
  // right-to-left to find ones that can be tiles together
  int input_idx = input->bounds.size() - 1;
  int output_idx = output->bounds.size() - 1;
  while ((input_idx >= 0) && (output_idx >= 0)) {
    std::vector<int> input_dims(1, input_idx);
    std::vector<int> output_dims(1, output_idx);
    size_t input_tile_volume = input->bounds[input_idx--];
    size_t output_tile_volume = output->bounds[output_idx--];
    while (input_tile_volume != output_tile_volume) {
      if (input_tile_volume < output_tile_volume) {
        input_dims.push_back(input_idx);
        input_tile_volume *= input->bounds[input_idx--];
      } else {
        output_dims.push_back(output_idx);
        output_tile_volume *= output->bounds[output_idx--];
      }
    }
    input_groups.emplace_back(input_dims);
    output_groups.emplace_back(output_dims);
  }
  // In order to use the output launch space, we need to make sure that
  // all but the earliest dimension in each output group has a partitioning
  // strategy of 1 or else we won't be able to compute a partition that
  // will allow for densely tiled copies. In the future we could fix this
  // by computing a generalized index launch space and then mapping that
  // onto the original output launch space or just by using affine indirect
  // copy launchers when they are available.
  for (unsigned g = 0; g < output_groups.size(); g++) {
    const std::vector<int>& input_group = input_groups[g];
    const std::vector<int>& output_group = output_groups[g];
    for (unsigned idx = 0; idx < (output_group.size() - 1); idx++)
      assert(strategy->dim[output_group[idx]] == 1);
    // the size of the earliest dimension in the input group must also
    // be divisible by the number of chunks
    assert(
        (input->bounds[input_group.back()] %
         strategy->dim[output_group.back()]) == 0);
    // the output bounds also need to be evenly divisible too or this will not
    // work
    assert(
        (output->bounds[output_group.back()] %
         strategy->dim[output_group.back()]) == 0);
  }
  inputs.push_back(input);
  outputs.push_back(output);
}

Domain
Reshape::GetInputBounds(Processor proc)
{
  const DomainPoint local_point = strategy->find_local_point(proc);
  DomainPoint lo, hi;
  const int input_dims = inputs[0]->bounds.size();
  lo.dim = input_dims;
  hi.dim = input_dims;
  for (unsigned g = 0; g < input_groups.size(); g++) {
    const std::vector<int>& input_group = input_groups[g];
    const std::vector<int>& output_group = output_groups[g];
    // Everything but the first dimension in the group is full size
    // Remember that dimensions are in reverse order
    for (unsigned idx = 0; idx < (input_group.size() - 1); idx++) {
      int dim = input_group[idx];
      lo[dim] = 0;
      hi[dim] = inputs[0]->bounds[dim] - 1;
    }
    // For the first dimension, divide it by the chunk of the
    // corresponding output dimension
    int input_dim = input_group.back();
    int output_dim = output_group.back();
    assert(output_dim < local_point.dim);
    assert(output_dim < strategy->nDims);
    size_t chunks = strategy->dim[output_dim];
    assert((inputs[0]->bounds[input_dim] % chunks) == 0);
    size_t chunk = inputs[0]->bounds[input_dim] / chunks;
    lo[input_dim] = local_point[output_dim] * chunk;
    hi[input_dim] = lo[input_dim] + chunk - 1;
  }
  return Domain(lo, hi);
}

Domain
Reshape::GetOutputBounds(Processor proc)
{
  assert(outputs[0]->bounds.size() == size_t(strategy->nDims));
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
Reshape::Load(Processor proc)
{
  assert(proc.kind() == strategy->kind);
  // If this processor is not used for this layer there is nothing to do
  if (!strategy->is_local_processor(proc))
    return;
  const unsigned local_index = strategy->find_local_offset(proc);
  ReshapeArgs& proc_args = args[local_index];
  proc_args.owner = this;
  proc_args.input_bounds = GetInputBounds(proc);
  proc_args.output_bounds = GetOutputBounds(proc);
  proc_args.datatype = outputs[0]->type;
  // volumes of the tiles should be the same
  assert(
      proc_args.input_bounds.get_volume() ==
      proc_args.output_bounds.get_volume());
  proc_args.copy_size =
      proc_args.input_bounds.get_volume() * sizeof_datatype(proc_args.datatype);
}

void
Reshape::initialize(
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
    values[itr.p] = TaskArgument(args + local_index, sizeof(ReshapeArgs));
  }
  argmaps[instance_index] = runtime->construct_future_map(
      ctx, launch_space, values, true /*collective*/, shardfn->sharding_id);

  IndexTaskLauncher& launcher = launchers[instance_index];
  launcher = IndexTaskLauncher(
      RESHAPE_TASK_ID, launch_space, TaskArgument(NULL, 0),
      ArgumentMap(argmaps[instance_index]), Predicate::TRUE_PRED,
      false /*must*/, mapper, strategy->tag);
  LogicalRegion input_region = inputs[0]->region[instance_index];
  assert(outputs.size() == 1);
  LogicalRegion output_region = instance->create_tensor_region(outputs[0]);

  // Create partitions for the regions
  DomainTransform transform;
  transform.m = inputs[0]->bounds.size();
  transform.n = outputs[0]->bounds.size();
  for (int i = 0; i < transform.m; i++)
    for (int j = 0; j < transform.n; j++)
      transform.matrix[i * transform.n + j] = 0;
  DomainPoint lo, hi;
  lo.dim = transform.m;
  hi.dim = transform.m;
  for (unsigned g = 0; g < input_groups.size(); g++) {
    const std::vector<int>& input_group = input_groups[g];
    const std::vector<int>& output_group = output_groups[g];
    // Everything but the first dimension in the group is full size
    // Remember that dimensions are in reverse order
    for (unsigned idx = 0; idx < (input_group.size() - 1); idx++) {
      int dim = input_group[idx];
      lo[dim] = 0;
      hi[dim] = inputs[0]->bounds[dim] - 1;
    }
    // For the first dimension, divide it by the chunk of the
    // corresponding output dimension
    int input_dim = input_group.back();
    int output_dim = output_group.back();
    assert(output_dim < strategy->nDims);
    size_t chunks = strategy->dim[output_dim];
    assert((inputs[0]->bounds[input_dim] % chunks) == 0);
    size_t chunk = inputs[0]->bounds[input_dim] / chunks;
    lo[input_dim] = 0;
    hi[input_dim] = chunk - 1;
    transform.matrix[input_dim * transform.n + output_dim] = 1;
  }
  Domain extent(lo, hi);
  IndexPartition index_part = instance->find_or_create_partition(
      input_region.get_index_space(), launch_space, transform, extent,
      LEGION_DISJOINT_COMPLETE_KIND);
  LogicalPartition input_part = runtime->get_logical_partition_by_tree(
      ctx, index_part, input_region.get_field_space(),
      input_region.get_tree_id());
  LogicalPartition output_part =
      instance->find_or_create_tiled_partition(outputs[0], strategy);
  launcher.add_region_requirement(RegionRequirement(
      input_part, 0 /*projection id*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE,
      input_region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(
      output_part, 0 /*projection id*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE,
      output_region));
  launcher.add_field(1, FID_DATA);
}

void
Reshape::forward(
    LegionModelInstance* instance, const unsigned instance_index,
    Runtime* runtime, Context ctx, MapperID mapper)
{
  runtime->execute_index_space(ctx, launchers[instance_index]);
}

void
Reshape::finalize(
    LegionModelInstance* instance, const unsigned instance_index,
    Runtime* runtime, Context ctx, MapperID mapper)
{
  argmaps[instance_index] = FutureMap();
}

void
Reshape::Free(Processor proc)
{
  // Nothing to do in this case
}

/*static*/ void
Reshape::PreregisterTaskVariants(void)
{
  {
    TaskVariantRegistrar cpu_registrar(RESHAPE_TASK_ID, "Reshape CPU");
    cpu_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    cpu_registrar.set_leaf();
    Runtime::preregister_task_variant<forward_cpu>(
        cpu_registrar, "Reshape Operator");
  }
#ifdef LEGION_USE_CUDA
  {
    TaskVariantRegistrar gpu_registrar(RESHAPE_TASK_ID, "Reshape GPU");
    gpu_registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    gpu_registrar.set_leaf();
    Runtime::preregister_task_variant<forward_gpu>(
        gpu_registrar, "Reshape Operator");
  }
#endif
}

/*static*/ void
Reshape::forward_cpu(
    const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
    Runtime* runtime)
{
  assert(task->local_arglen == sizeof(ReshapeArgs));
  const ReshapeArgs* args = (const ReshapeArgs*)task->local_args;
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const void* input_ptr = nullptr;
  void* output_ptr = nullptr;
  size_t volume = 0;
  switch (args->input_bounds.get_dim()) {
#define DIMFUNC(DIM)                                           \
  case DIM: {                                                  \
    const Rect<DIM> bounds = args->input_bounds;               \
    volume = bounds.volume();                                  \
    input_ptr = TensorAccessor<LEGION_READ_ONLY, DIM>::access( \
        args->datatype, bounds, regions[0]);                   \
    break;                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      abort();
  }
  switch (args->output_bounds.get_dim()) {
#define DIMFUNC(DIM)                                                \
  case DIM: {                                                       \
    const Rect<DIM> bounds = args->output_bounds;                   \
    output_ptr = TensorAccessor<LEGION_WRITE_DISCARD, DIM>::access( \
        args->datatype, bounds, regions[1]);                        \
    break;                                                          \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      abort();
  }
  memcpy(output_ptr, input_ptr, args->copy_size);
}

ReshapeArgs::ReshapeArgs(void) {}

#ifdef LEGION_USE_CUDA
/*static*/ void
Reshape::forward_gpu(
    const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
    Runtime* runtime)
{
  assert(task->local_arglen == sizeof(ReshapeArgs));
  const ReshapeArgs* args = (const ReshapeArgs*)task->local_args;
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
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const void* input_ptr = nullptr;
  void* output_ptr = nullptr;
  size_t volume = 0;
  switch (args->input_bounds.get_dim()) {
#define DIMFUNC(DIM)                                           \
  case DIM: {                                                  \
    const Rect<DIM> bounds = args->input_bounds;               \
    volume = bounds.volume();                                  \
    input_ptr = TensorAccessor<LEGION_READ_ONLY, DIM>::access( \
        args->datatype, bounds, regions[0]);                   \
    break;                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      abort();
  }
  switch (args->output_bounds.get_dim()) {
#define DIMFUNC(DIM)                                                \
  case DIM: {                                                       \
    const Rect<DIM> bounds = args->output_bounds;                   \
    output_ptr = TensorAccessor<LEGION_WRITE_DISCARD, DIM>::access( \
        args->datatype, bounds, regions[1]);                        \
    break;                                                          \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      abort();
  }
#ifdef DISABLE_LEGION_CUDA_HIJACK
  CHECK_CUDA(cudaMemcpyAsync(
      output_ptr, input_ptr, args->copy_size, cudaMemcpyDeviceToDevice));
#else
  CHECK_CUDA(cudaMemcpyAsync(
      output_ptr, input_ptr, args->copy_size, cudaMemcpyDeviceToDevice,
      stream));
#endif
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
        "%s [Reshape] forward time (CF) = %.2fms\n",
        args->owner->op_name.c_str(), elapsed);
  }
}
#endif

}}}  // namespace triton::backend::legion

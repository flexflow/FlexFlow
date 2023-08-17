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

#include "matmul.h"

using namespace Legion;

namespace triton { namespace backend { namespace legion {

MatMulProjectionFunctor::MatMulProjectionFunctor(
    ProjectionID id, const DomainTransform& trans)
    : ProjectionFunctor(), functor_id(id), domain_transform(trans)
{
}

LogicalRegion
MatMulProjectionFunctor::project(
    LogicalPartition upper_bound, const DomainPoint& point,
    const Domain& domain)
{
  return runtime->get_logical_subregion_by_color(upper_bound, transform(point));
}

MatMulArgs::MatMulArgs(void) {}

MatMul::MatMul(
    LegionModelState* model, const LayerStrategy* strategy, const char* name)
    : Operator(model, strategy, OperatorType::OP_MATMUL, name, 2, 0, 1),
      in1_proj(nullptr), in2_proj(nullptr)
{
}

template <unsigned DIM>
void
MatMul::compute_in1_parameters(Tensor* in1, Tensor* out)
{
  Rect<DIM> extent, colors;
  Transform<DIM, DIM> transform;
  for (int i = 0; i < DIM; i++)
    for (int j = 0; j < DIM; j++) transform[i][j] = 0;
  assert(out->bounds.size() >= in1->bounds.size());
  size_t dimoff = out->bounds.size() - in1->bounds.size();
  for (int i = 0; i < DIM; i++) {
    extent.lo[i] = 0;
    colors.lo[i] = 0;
    if (i == (DIM - 1)) {
      /* need the whole dimension */
      extent.hi[i] = in1->bounds[i] - 1; /*inclusive*/
      colors.hi[i] = 0;
    } else if (in1->bounds[i] == 1) {
      extent.hi[i] = 0;
      colors.hi[i] = 0;
    } else {
      size_t pieces = strategy->dim[dimoff + 1];
      size_t chunks = (in1->bounds[i] + pieces - 1) / pieces;
      extent.hi[i] = chunks - 1; /*inclusive*/
      colors.hi[i] = pieces - 1; /*inclusive*/
    }
  }
  in1_transform = transform;
  in1_extent = extent;
  in1_colors = colors;
}

template <unsigned DIM>
void
MatMul::compute_in2_parameters(Tensor* in2, Tensor* out)
{
  Rect<DIM> extent, colors;
  Transform<DIM, DIM> transform;
  for (int i = 0; i < DIM; i++)
    for (int j = 0; j < DIM; j++) transform[i][j] = 0;
  assert(out->bounds.size() >= in2->bounds.size());
  size_t dimoff = out->bounds.size() - in2->bounds.size();
  for (int i = 0; i < DIM; i++) {
    extent.lo[i] = 0;
    colors.lo[i] = 0;
    if (i == (DIM - 2)) {
      /* need the whole dimension */
      extent.hi[i] = in2->bounds[i] - 1; /*inclusive*/
      colors.hi[i] = 0;
    } else if (in2->bounds[i] == 1) {
      extent.hi[i] = 0;
      colors.hi[i] = 0;
    } else {
      size_t pieces = strategy->dim[dimoff + i];
      size_t chunks = (in2->bounds[i] + pieces - 1) / pieces;
      extent.hi[i] = chunks - 1; /*inclusive*/
      colors.hi[i] = pieces - 1; /*inclusive*/
    }
  }
  in2_transform = transform;
  in2_extent = extent;
  in2_colors = colors;
}

void
MatMul::Configure(Tensor* in1, Tensor* in2, Tensor* out)
{
  assert(in1 != nullptr);
  assert(in2 != nullptr);
  assert(out != nullptr);
  inputs.push_back(in1);
  inputs.push_back(in2);
  outputs.push_back(out);

  if ((in1->bounds.size() == 1) && (in2->bounds.size() == 1)) {
    fprintf(stderr, "TODO: support for dot-product in matmul operator");
    abort();
  } else if (in1->bounds.size() == 1) {
    const size_t in2_dim = in2->bounds.size();
    const size_t out_dim = out->bounds.size();
    assert(in2_dim >= 2);
    assert(out_dim >= 1);
    const size_t n = out->bounds[out_dim - 1];
    const size_t k = in1->bounds[0];
    assert(in2->bounds[in2_dim - 2] == k);
    assert(in2->bounds[in2_dim - 1] == n);
    // make sure all the other dimensions align or broadcast
    unsigned in2_broadcasts = 0;
    for (unsigned off = 3; off <= out_dim; off++) {
      const size_t out_size = out->bounds[out_dim - off];
      if (off <= in2_dim) {
        const size_t size = in2->bounds[in2_dim - off];
        assert((size == 1) || (size == out_size));
        if (size == 1)
          in2_broadcasts |= (1 << (off - 3));
      }
    }
    FunctorKey in1_key(out->bounds.size(), 1, 0);
    FunctorTable::const_iterator finder = in1_functors.find(in1_key);
    assert(finder != in1_functors.end());
    in1_proj = finder->second;

    FunctorKey in2_key(out->bounds.size(), in2->bounds.size(), in2_broadcasts);
    finder = in2_functors.find(in2_key);
    assert(finder != in2_functors.end());
    in2_proj = finder->second;

    Rect<1> extent, colors;
    Transform<1, 1> transform;
    transform[0][0] = 0;
    extent.lo[0] = 0;
    extent.hi[0] = in1->bounds[0] - 1;  // inclusive
    colors.lo[0] = 0;
    colors.hi[0] = 0;
    in1_transform = transform;
    in1_extent = extent;
    in1_colors = colors;

    switch (in2->bounds.size()) {
#define DIMFUNC(DIM)                       \
  case DIM: {                              \
    compute_in2_parameters<DIM>(in2, out); \
    break;                                 \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        abort();
    }
  } else if (in2->bounds.size() == 1) {
    const size_t in1_dim = in1->bounds.size();
    const size_t out_dim = out->bounds.size();
    assert(in1_dim >= 2);
    const size_t m = (out_dim > 1) ? out->bounds[out_dim - 2] : 1;
    assert(out->bounds[out_dim - 1] == 1);
    assert(in1->bounds[in1_dim - 2] == m);
    const size_t k = in1->bounds[in1_dim - 1];
    assert(in2->bounds[in2->bounds[0]] == k);
    // make sure all the other dimensions align or broadcast
    unsigned in1_broadcasts = 0;
    for (unsigned off = 3; off <= out_dim; off++) {
      const size_t out_size = out->bounds[out_dim - off];
      if (off <= in1_dim) {
        const size_t size = in1->bounds[in1_dim - off];
        assert((size == 1) || (size == out_size));
        if (size == 1)
          in1_broadcasts |= (1 << (off - 3));
      }
    }
    FunctorKey in1_key(out->bounds.size(), in1->bounds.size(), in1_broadcasts);
    FunctorTable::const_iterator finder = in1_functors.find(in1_key);
    assert(finder != in1_functors.end());
    in1_proj = finder->second;

    FunctorKey in2_key(out->bounds.size(), 1, 0);
    finder = in2_functors.find(in2_key);
    assert(finder != in2_functors.end());
    in2_proj = finder->second;

    switch (in1->bounds.size()) {
#define DIMFUNC(DIM)                       \
  case DIM: {                              \
    compute_in1_parameters<DIM>(in1, out); \
    break;                                 \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        abort();
    }

    Rect<1> extent, colors;
    Transform<1, 1> transform;
    transform[0][0] = 0;
    extent.lo[0] = 0;
    extent.hi[0] = in2->bounds[0] - 1;  // inclusive
    colors.lo[0] = 0;
    colors.hi[0] = 0;
    in2_transform = transform;
    in2_extent = extent;
    in2_colors = colors;

  } else {
    // all tensors have at least two dimensions
    const size_t in1_dim = in1->bounds.size();
    const size_t in2_dim = in2->bounds.size();
    const size_t out_dim = out->bounds.size();
    assert(in1_dim >= 2);
    assert(in2_dim >= 2);
    assert(out_dim >= 2);
    const size_t m = out->bounds[out_dim - 2];
    const size_t n = out->bounds[out_dim - 1];
    assert(in1->bounds[in1_dim - 2] == m);
    const size_t k = in1->bounds[in1_dim - 1];
    assert(in2->bounds[in2_dim - 2] == k);
    assert(in2->bounds[in2_dim - 1] == n);
    // make sure all the other dimensions align or can broadcast
    unsigned in1_broadcasts = 0, in2_broadcasts = 0;
    for (unsigned off = 3; off <= out_dim; off++) {
      const size_t out_size = out->bounds[out_dim - off];
      if (off <= in1_dim) {
        const size_t size = in1->bounds[in1_dim - off];
        assert((size == 1) || (size == out_size));
        if (size == 1)
          in1_broadcasts |= (1 << (off - 3));
      }
      if (off <= in2_dim) {
        const size_t size = in2->bounds[in2_dim - off];
        assert((size == 1) || (size == out_size));
        if (size == 1)
          in2_broadcasts |= (1 << (off - 3));
      }
    }
    FunctorKey in1_key(out->bounds.size(), in1->bounds.size(), in1_broadcasts);
    FunctorTable::const_iterator finder = in1_functors.find(in1_key);
    assert(finder != in1_functors.end());
    in1_proj = finder->second;

    FunctorKey in2_key(out->bounds.size(), in2->bounds.size(), in2_broadcasts);
    finder = in2_functors.find(in2_key);
    assert(finder != in2_functors.end());
    in2_proj = finder->second;

    // Finally fill in the input transforms, extents, and colors for the inputs
    switch (in1->bounds.size()) {
#define DIMFUNC(DIM)                       \
  case DIM: {                              \
    compute_in1_parameters<DIM>(in1, out); \
    break;                                 \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        abort();
    }

    switch (in2->bounds.size()) {
#define DIMFUNC(DIM)                       \
  case DIM: {                              \
    compute_in2_parameters<DIM>(in2, out); \
    break;                                 \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        abort();
    }
  }
}

Domain
MatMul::GetIn1Bounds(Processor proc)
{
  assert(in1_proj != nullptr);
  const DomainPoint point = strategy->find_local_point(proc);
  const DomainPoint local = in1_proj->transform(point);
  const DomainPoint offset = in1_transform * local;
  switch (inputs[0]->bounds.size()) {
#define DIMFUNC(DIM)                                                   \
  case DIM: {                                                          \
    Point<DIM> off = offset;                                           \
    Rect<DIM> extent = in1_extent;                                     \
    Rect<DIM> bounds(extent.lo + off, extent.hi + off);                \
    Point<DIM> upper;                                                  \
    for (int i = 0; i < DIM; i++) upper[i] = inputs[0]->bounds[i] - 1; \
    Rect<DIM> full(Point<DIM>::ZEROES(), upper);                       \
    Rect<DIM> result = full.intersection(bounds);                      \
    return Domain(result);                                             \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      abort();
  }
  return Domain();
}

Domain
MatMul::GetIn2Bounds(Processor proc)
{
  assert(in2_proj != nullptr);
  const DomainPoint point = strategy->find_local_point(proc);
  const DomainPoint local = in2_proj->transform(point);
  const DomainPoint offset = in2_transform * local;
  switch (inputs[1]->bounds.size()) {
#define DIMFUNC(DIM)                                                   \
  case DIM: {                                                          \
    Point<DIM> off = offset;                                           \
    Rect<DIM> extent = in2_extent;                                     \
    Rect<DIM> bounds(extent.lo + off, extent.hi + off);                \
    Point<DIM> upper;                                                  \
    for (int i = 0; i < DIM; i++) upper[i] = inputs[1]->bounds[i] - 1; \
    Rect<DIM> full(Point<DIM>::ZEROES(), upper);                       \
    Rect<DIM> result = full.intersection(bounds);                      \
    return Domain(result);                                             \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      abort();
  }
  return Domain();
}

Domain
MatMul::GetOutBounds(Processor proc)
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
MatMul::Load(Processor proc)
{
  assert(proc.kind() == strategy->kind);
  // If this processor is not used for this layer there is nothing to do
  if (!strategy->is_local_processor(proc))
    return;
  const unsigned local_index = strategy->find_local_offset(proc);
  MatMulArgs& proc_args = args[local_index];
  proc_args.owner = this;
  proc_args.in1_bounds = GetIn1Bounds(proc);
  proc_args.in2_bounds = GetIn2Bounds(proc);
  proc_args.out_bounds = GetOutBounds(proc);
  proc_args.in1_datatype = inputs[0]->type;
  proc_args.in2_datatype = inputs[1]->type;
  proc_args.out_datatype = outputs[0]->type;
#ifdef LEGION_USE_CUDA
  if (proc.kind() == Processor::TOC_PROC)
    proc_args.cublas = model->runtime_->cublas[local_index];
#endif
}

void
MatMul::initialize(
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
    values[itr.p] = TaskArgument(args + local_index, sizeof(MatMulArgs));
  }
  argmaps[instance_index] = runtime->construct_future_map(
      ctx, launch_space, values, true /*collective*/, shardfn->sharding_id);

  IndexTaskLauncher& launcher = launchers[instance_index];
  launcher = IndexTaskLauncher(
      MATMUL_TASK_ID, launch_space, TaskArgument(NULL, 0),
      ArgumentMap(argmaps[instance_index]), Predicate::TRUE_PRED,
      false /*must*/, mapper, strategy->tag);
  // Create partition for the output region
  LogicalRegion output_region = instance->create_tensor_region(outputs[0]);
  LogicalPartition output_part =
      instance->find_or_create_tiled_partition(outputs[0], strategy);
  launcher.add_region_requirement(RegionRequirement(
      output_part, 0 /*projection id*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE,
      output_region));
  launcher.add_field(0, FID_DATA);
  // Create partition for the input regions
  LogicalRegion in1_region = inputs[0]->region[instance_index];
  IndexSpace in1_colorspace = instance->find_or_create_index_space(in1_colors);
  IndexPartition index1_part = instance->find_or_create_partition(
      in1_region.get_index_space(), in1_colorspace, in1_transform, in1_extent,
      LEGION_DISJOINT_COMPLETE_KIND);
  LogicalPartition in1_part = runtime->get_logical_partition_by_tree(
      ctx, index1_part, in1_region.get_field_space(), in1_region.get_tree_id());
  launcher.add_region_requirement(RegionRequirement(
      in1_part, in1_proj->functor_id, LEGION_READ_ONLY, LEGION_EXCLUSIVE,
      in1_region));
  launcher.add_field(1, FID_DATA);
  LogicalRegion in2_region = inputs[1]->region[instance_index];
  IndexSpace in2_colorspace = instance->find_or_create_index_space(in2_colors);
  IndexPartition index2_part = instance->find_or_create_partition(
      in2_region.get_index_space(), in2_colorspace, in2_transform, in2_extent,
      LEGION_DISJOINT_COMPLETE_KIND);
  LogicalPartition in2_part = runtime->get_logical_partition_by_tree(
      ctx, index2_part, in2_region.get_field_space(), in2_region.get_tree_id());
  launcher.add_region_requirement(RegionRequirement(
      in2_part, in2_proj->functor_id, LEGION_READ_ONLY, LEGION_EXCLUSIVE,
      in2_region));
  launcher.add_field(2, FID_DATA);
}

void
MatMul::forward(
    LegionModelInstance* instance, const unsigned instance_index,
    Runtime* runtime, Context ctx, MapperID mapper)
{
  runtime->execute_index_space(ctx, launchers[instance_index]);
}

void
MatMul::finalize(
    LegionModelInstance* instance, const unsigned instance_index,
    Runtime* runtime, Context ctx, MapperID mapper)
{
  argmaps[instance_index] = FutureMap();
}

void
MatMul::Free(Processor proc)
{
  // Nothing to do in this case
}

/*static instantiations*/
MatMul::FunctorTable MatMul::in1_functors;
MatMul::FunctorTable MatMul::in2_functors;

template <unsigned IDIM, unsigned ODIM>
/*static*/ void
MatMul::generate_specific_functors(void)
{
  assert(ODIM <= IDIM);
  // Enumerate all the combinations of bit masks for broadcasting
  const unsigned combinations = (ODIM >= 2) ? 1 << (ODIM - 2) : 1;
  for (unsigned idx = 0; idx < combinations; idx++) {
    const FunctorKey key(IDIM, ODIM, idx);
    // Input1 case: partition on ODIM-2 but broadcast on ODIM-1
    {
      Transform<ODIM, IDIM> transform;
      // Initialize everything to zeros to start
      for (int i = 0; i < ODIM; i++)
        for (int j = 0; j < IDIM; j++) transform[i][j] = 0;
      // Work backwards for broadcasting
      for (int off = 1; off <= ODIM; off++) {
        if (off == 1) {
          // broadcast
          transform[ODIM - off][IDIM - off] = 0;
        } else if (off == 2) {
          // use partition
          transform[ODIM - off][IDIM - off] = 1;
        } else {
          // check for broadcast
          transform[ODIM - off][IDIM - off] = (idx & (1 << (idx - 3))) ? 0 : 1;
        }
      }
      DomainTransform domain_transform(transform);
      ProjectionID id = Runtime::generate_static_projection_id();
      MatMulProjectionFunctor* functor =
          new MatMulProjectionFunctor(id, domain_transform);
      Runtime::preregister_projection_functor(id, functor);
      assert(in1_functors.find(key) == in1_functors.end());
      in1_functors[key] = functor;
    }
    // Input2 case: broadcast on ODIM-2 but partition on ODIM-1
    {
      Transform<ODIM, IDIM> transform;
      // Initialize everything to zeros to start
      for (int i = 0; i < ODIM; i++)
        for (int j = 0; j < IDIM; j++) transform[i][j] = 0;
      // Work backwards for broadcasting
      for (int off = 1; off <= ODIM; off++) {
        if (off == 1) {
          // use partition (unless we're a vector so we're broadcasting)
          transform[ODIM - off][IDIM - off] = (ODIM == 1) ? 0 : 1;
        } else if (off == 2) {
          // broadcast
          transform[ODIM - off][IDIM - off] = 0;
        } else {
          // check for broadcast
          transform[ODIM - off][IDIM - off] = (idx & (1 << (idx - 3))) ? 0 : 1;
        }
      }
      DomainTransform domain_transform(transform);
      ProjectionID id = Runtime::generate_static_projection_id();
      MatMulProjectionFunctor* functor =
          new MatMulProjectionFunctor(id, domain_transform);
      Runtime::preregister_projection_functor(id, functor);
      assert(in2_functors.find(key) == in2_functors.end());
      in2_functors[key] = functor;
    }
  }
}

template <unsigned IDIM>
/*static*/ void
MatMul::generate_all_functors(void)
{
  for (int i = 1; i <= IDIM; i++) {
    switch (i) {
#define DIMFUNC(DIM)                         \
  case DIM: {                                \
    generate_specific_functors<IDIM, DIM>(); \
    break;                                   \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        abort();
    }
  }
}

/*static*/ void
MatMul::PreregisterTaskVariants(void)
{
  // Create all possible functors we might need here so these data
  // structures can be read-only after this point
  for (int i = 2; i <= LEGION_MAX_DIM; i++) {
    switch (i) {
#define DIMFUNC(DIM)              \
  case DIM: {                     \
    generate_all_functors<DIM>(); \
    break;                        \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        abort();
    }
  }
  {
    TaskVariantRegistrar cpu_registrar(MATMUL_TASK_ID, "MatMul CPU");
    cpu_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    cpu_registrar.set_leaf();
    Runtime::preregister_task_variant<forward_cpu>(
        cpu_registrar, "MatMul Operator");
  }
#ifdef LEGION_USE_CUDA
  {
    TaskVariantRegistrar gpu_registrar(MATMUL_TASK_ID, "MatMul GPU");
    gpu_registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    gpu_registrar.set_leaf();
    Runtime::preregister_task_variant<forward_gpu>(
        gpu_registrar, "MatMul Operator");
  }
#endif
}

/*static*/ void
MatMul::forward_cpu(
    const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
    Runtime* runtime)
{
  // TODO: implement this with OpenBLAS or something like it
  abort();
}

#ifdef LEGION_USE_CUDA
/*static*/ void
MatMul::forward_gpu(
    const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
    Runtime* runtime)
{
  assert(task->local_arglen == sizeof(MatMulArgs));
  const MatMulArgs* args = (const MatMulArgs*)task->local_args;
#ifndef DISABLE_LEGION_CUDA_HIJACK
  ::cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUBLAS(cublasSetStream(args->cublas, stream));
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
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  // cublas is dumb and doesn't support row-major, so reverse the matrix
  // order to help cublas think things are column-major
  // effectively we get NxM = NxK * KxM
  uint8_t* out_ptr = nullptr;
  size_t m, n, k, batch_count = 1;
  size_t lda, ldb, ldc, astride, bstride, cstride;
  switch (args->out_bounds.get_dim()) {
#define DIMFUNC(DIM)                                                       \
  case DIM: {                                                              \
    const Rect<DIM> bounds = args->out_bounds;                             \
    out_ptr = (uint8_t*)TensorAccessor<LEGION_WRITE_DISCARD, DIM>::access( \
        args->out_datatype, bounds, regions[0]);                           \
    for (int i = 0; i < (DIM - 2); i++)                                    \
      batch_count *= ((bounds.hi[i] - bounds.lo[i]) + 1);                  \
    if (DIM == 1) {                                                        \
      assert(                                                              \
          (args->in1_bounds.get_dim() == 1) ||                             \
          (args->in2_bounds.get_dim() == 1));                              \
      if (args->in1_bounds.get_dim() == 1) {                               \
        n = 1;                                                             \
        m = (bounds.hi[0] - bounds.lo[0]) + 1;                             \
      } else {                                                             \
        n = (bounds.hi[0] - bounds.lo[0]) + 1;                             \
        m = 1;                                                             \
      }                                                                    \
    } else {                                                               \
      n = (bounds.hi[DIM - 2] - bounds.lo[DIM - 2]) + 1;                   \
      m = (bounds.hi[DIM - 1] - bounds.lo[DIM - 1]) + 1;                   \
    }                                                                      \
    ldc = m;                                                               \
    cstride = m * n;                                                       \
    break;                                                                 \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      abort();
  }
  bool has_broadcast = false;
  const uint8_t *in1_ptr = nullptr, *in2_ptr = nullptr;
  switch (args->in1_bounds.get_dim()) {
#define DIMFUNC(DIM)                                                         \
  case DIM: {                                                                \
    const Rect<DIM> bounds = args->in1_bounds;                               \
    in1_ptr = (const uint8_t*)TensorAccessor<LEGION_READ_ONLY, DIM>::access( \
        args->in1_datatype, bounds, regions[1]);                             \
    k = (bounds.hi[DIM - 1] - bounds.lo[DIM - 1]) + 1;                       \
    ldb = (DIM == 1) ? 1 : k;                                                \
    if (DIM == 1)                                                            \
      bstride = k;                                                           \
    else                                                                     \
      bstride = k * ((bounds.hi[DIM - 2] - bounds.lo[DIM - 2]) + 1);         \
    if (!has_broadcast) {                                                    \
      if (DIM == args->out_bounds.get_dim()) {                               \
        const Rect<DIM> out_bounds = args->out_bounds;                       \
        for (int i = 0; i < (DIM - 2); i++) {                                \
          if ((bounds.hi[i] > 0) || (out_bounds.hi[i] == 0))                 \
            continue;                                                        \
          has_broadcast = true;                                              \
          break;                                                             \
        }                                                                    \
      } else {                                                               \
        has_broadcast = true;                                                \
      }                                                                      \
    }                                                                        \
    break;                                                                   \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      abort();
  }
  switch (args->in2_bounds.get_dim()) {
#define DIMFUNC(DIM)                                                         \
  case DIM: {                                                                \
    const Rect<DIM> bounds = args->in2_bounds;                               \
    in2_ptr = (const uint8_t*)TensorAccessor<LEGION_READ_ONLY, DIM>::access( \
        args->in2_datatype, bounds, regions[2]);                             \
    lda = (bounds.hi[DIM - 1] - bounds.lo[DIM - 1]) + 1;                     \
    if (DIM == 1)                                                            \
      astride = lda;                                                         \
    else                                                                     \
      astride = lda * ((bounds.hi[DIM - 2] - bounds.lo[DIM - 2]) + 1);       \
    if (!has_broadcast) {                                                    \
      if (DIM == args->out_bounds.get_dim()) {                               \
        const Rect<DIM> out_bounds = args->out_bounds;                       \
        for (int i = 0; i < (DIM - 2); i++) {                                \
          if ((bounds.hi[i] > 0) || (out_bounds.hi[i] == 0))                 \
            continue;                                                        \
          has_broadcast = true;                                              \
          break;                                                             \
        }                                                                    \
      } else {                                                               \
        has_broadcast = true;                                                \
      }                                                                      \
    }                                                                        \
    break;                                                                   \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      abort();
  }
  if (has_broadcast) {
    assert(args->out_bounds.get_dim() > 2);
    std::vector<size_t> in1_iterations(args->out_bounds.get_dim() - 2, 1);
    std::vector<size_t> in2_iterations(args->out_bounds.get_dim() - 2, 1);
    std::vector<size_t> out_iterations(args->out_bounds.get_dim() - 2);
    const int in1_dims = args->in1_bounds.get_dim();
    for (int off = 0; off < (args->in1_bounds.get_dim() - 2); off++)
      in1_iterations[in1_iterations.size() - off] =
          (args->in1_bounds.hi()[in1_dims - (off + 2)] -
           args->in1_bounds.lo()[in1_dims - (off + 2)]) +
          1;
    const int in2_dims = args->in2_bounds.get_dim();
    for (int off = 0; off < (args->in2_bounds.get_dim() - 2); off++)
      in2_iterations[in2_iterations.size() - off] =
          (args->in2_bounds.hi()[in2_dims - (off + 2)] -
           args->in2_bounds.lo()[in2_dims - (off + 2)]) +
          1;
    for (unsigned dim = 0; dim < out_iterations.size(); dim++)
      out_iterations[dim] =
          (args->out_bounds.hi()[dim] - args->out_bounds.lo()[dim]) + 1;
    // Find the "last full dim" without a broadcast
    int last_full_dim = in1_iterations.size();
    size_t partial_batch_count = 1;
    for (int idx = in1_iterations.size() - 1; idx >= 0; --idx) {
      if (in1_iterations[idx] == in2_iterations[idx]) {
        last_full_dim = idx;
        partial_batch_count *= in1_iterations[idx];
        continue;
      }
      assert((in1_iterations[idx] == 1) || (in2_iterations[idx] == 1));
      break;
    }
    assert(last_full_dim > 0);
    assert((batch_count % partial_batch_count) == 0);
    std::vector<size_t> in1_indexes(args->out_bounds.get_dim() - 2, 0);
    std::vector<size_t> in2_indexes(args->out_bounds.get_dim() - 2, 0);
    std::vector<size_t> out_indexes(args->out_bounds.get_dim() - 2, 0);
    while (batch_count > 0) {
      // iterate the loops
      for (int dim = last_full_dim - 1; dim >= 0; dim++) {
        if (++out_indexes[dim] < out_iterations[dim]) {
          // step the in1 and in2 indexes while checking for broadcasting
          if (in1_iterations[dim] > 1) {
            ++in1_indexes[dim];
            assert(in1_indexes[dim] == out_indexes[dim]);
          }
          if (in2_iterations[dim] > 1) {
            ++in2_indexes[dim];
            assert(in2_indexes[dim] == out_indexes[dim]);
          }
          break;
        } else {
          // reset and ripple carry over to the next dim
          in1_indexes[dim] = 0;
          in2_indexes[dim] = 0;
          out_indexes[dim] = 0;
          assert(dim > 0);
        }
      }
      // compute the local pointers based on our indexes
      size_t in1_offset = in1_indexes[0];
      size_t in2_offset = in2_indexes[0];
      size_t out_offset = out_indexes[0];
      for (int dim = 1; dim < last_full_dim; dim++) {
        in1_offset = in1_offset * in1_iterations[dim] + in1_indexes[dim];
        in2_offset = in2_offset * in2_iterations[dim] + in2_indexes[dim];
        out_offset = out_offset * out_iterations[dim] + out_indexes[dim];
      }
      in1_offset *=
          partial_batch_count * bstride * sizeof_datatype(args->in1_datatype);
      in2_offset *=
          partial_batch_count * astride * sizeof_datatype(args->in2_datatype);
      out_offset *=
          partial_batch_count * cstride * sizeof_datatype(args->out_datatype);
      const uint8_t* in1_local = in1_ptr + in1_offset;
      const uint8_t* in2_local = in2_ptr + in2_offset;
      uint8_t* out_local = out_ptr + out_offset;
      switch (args->out_datatype) {
        // Use 32-bit intermediate for 16-bit float
        case DT_HALF:
        case DT_FLOAT: {
          float alpha = 1.f, beta = 0.f;
          CHECK_CUBLAS(cublasGemmStridedBatchedEx(
              args->cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
              in2_local, to_cuda_datatype(args->in2_datatype), lda, astride,
              in1_local, to_cuda_datatype(args->in1_datatype), ldb, bstride,
              &beta, out_local, to_cuda_datatype(args->out_datatype), ldc,
              cstride, partial_batch_count, CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
          break;
        }
        case DT_DOUBLE: {
          double alpha = 1.0, beta = 0.0;
          CHECK_CUBLAS(cublasGemmStridedBatchedEx(
              args->cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
              in2_local, to_cuda_datatype(args->in2_datatype), lda, astride,
              in1_local, to_cuda_datatype(args->in1_datatype), ldb, bstride,
              &beta, out_local, to_cuda_datatype(DT_DOUBLE), ldc, cstride,
              partial_batch_count, CUBLAS_COMPUTE_64F,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
          break;
        }
        case DT_INT32: {
          int32_t alpha = 1, beta = 0;
          CHECK_CUBLAS(cublasGemmStridedBatchedEx(
              args->cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, in2_ptr,
              to_cuda_datatype(args->in2_datatype), lda, astride, in1_ptr,
              to_cuda_datatype(args->in1_datatype), ldb, bstride, &beta,
              out_ptr, to_cuda_datatype(DT_INT32), ldc, cstride,
              partial_batch_count, CUBLAS_COMPUTE_32I,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
          break;
        }
        default:
          fprintf(
              stderr, "Unsupported cublas type for matmul %d\n",
              args->out_datatype);
          abort();
      }
      batch_count -= partial_batch_count;
    }
  } else {
    // This is the easy case where there are no broadcasts
    // so we can do the full batch matmul in a single call
    switch (args->out_datatype) {
      // Use 32-bit intermediate for 16-bit float
      case DT_HALF:
      case DT_FLOAT: {
        float alpha = 1.f, beta = 0.f;
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            args->cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, in2_ptr,
            to_cuda_datatype(args->in2_datatype), lda, astride, in1_ptr,
            to_cuda_datatype(args->in1_datatype), ldb, bstride, &beta, out_ptr,
            to_cuda_datatype(args->out_datatype), ldc, cstride, batch_count,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        break;
      }
      case DT_DOUBLE: {
        double alpha = 1.0, beta = 0.0;
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            args->cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, in2_ptr,
            to_cuda_datatype(args->in2_datatype), lda, astride, in1_ptr,
            to_cuda_datatype(args->in1_datatype), ldb, bstride, &beta, out_ptr,
            to_cuda_datatype(DT_DOUBLE), ldc, cstride, batch_count,
            CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        break;
      }
      case DT_INT32: {
        int32_t alpha = 1, beta = 0;
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            args->cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, in2_ptr,
            to_cuda_datatype(args->in2_datatype), lda, astride, in1_ptr,
            to_cuda_datatype(args->in1_datatype), ldb, bstride, &beta, out_ptr,
            to_cuda_datatype(DT_INT32), ldc, cstride, batch_count,
            CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        break;
      }
      default:
        fprintf(
            stderr, "Unsupported cublas type for matmul %d\n",
            args->out_datatype);
        abort();
    }
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
        "%s [MatMul] forward time = %.2fms\n", args->owner->op_name.c_str(),
        elapsed);
  }
}
#endif

}}}  // namespace triton::backend::legion

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

#include "model.h"
#include "cuda_helper.h"

Tensor FFModel::transpose(const Tensor& input,
                          const std::vector<int>& perm,
                          const char* name)
{
  Transpose* transpose = new Transpose(*this, input, perm, name);
  layers.push_back(transpose);
  return transpose->outputs[0];
}

Transpose::Transpose(FFModel& model,
                     const Tensor& input,
                     const std::vector<int>& _perm,
                     const char* name)
: Op(model, OP_TRANSPOSE, name, input)
{
  assert(_perm.size() == input.numDim);
  // Use Legion indexing to store perm
  for (int i = 0; i < input.numDim; i++)
    perm[i] = input.numDim - 1 - _perm[input.numDim - 1 - i];
  outputs[0].numDim = input.numDim;
  for (int i = 0; i < outputs[0].numDim; i++)
    outputs[0].adim[i] = input.adim[perm[i]];
  numOutputs = 1;
  numWeights = 0;
}

void Transpose::create_weights(FFModel& model)
{
  // Do nothing
}

void Transpose::create_output_and_partition(FFModel& model)
{
  int dim = inputs[0].numDim;
  switch (dim) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      task_is = model.get_or_create_task_is(DIM, name); \
      create_output_and_partition_with_dim<DIM>(model); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
    {
      // Unsupported dim for Transpose operator
      assert(false);
    }
  }
}

template<int NDIM>
void Transpose::create_output_and_partition_with_dim(FFModel& model)
{
  // Retrive the task indexspace for the op
  task_is = IndexSpaceT<NDIM>(model.get_or_create_task_is(NDIM, name));
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  // Current require all dimensions being transposed should not be partitioned
  for (int i = 0; i < NDIM; i++)
    if (i != perm[i])
      assert(part_rect.hi[i] == part_rect.lo[i]);
  int dims[NDIM];
  for (int i = 0; i < NDIM; i++)
    dims[i] = outputs[0].adim[NDIM-1-i];
  outputs[0] = model.create_tensor<NDIM>(dims, DT_FLOAT, this);
  outputs[0].owner_op = this;
  outputs[0].owner_idx = 0;
  Rect<NDIM> input_rect;
  input_rect = runtime->get_index_partition_color_space(
        ctx, inputs[0].part.get_index_partition());
  if (input_rect == part_rect) {
    input_lps[0] = inputs[0].part;
    input_grad_lps[0] = inputs[0].part_grad;
  } else {
    model.create_disjoint_partition<NDIM>(
        inputs[0], IndexSpaceT<NDIM>(task_is), input_lps[0], input_grad_lps[0]);
  }
}

void Transpose::init_meta(TransposeMeta *m, Domain const &in_domain, Domain const &out_domain) const
{
  for (int i = 0; i < out_domain.get_dim(); i++) {
    assert(out_domain.hi()[i] == in_domain.hi()[this->perm[i]]);
    assert(out_domain.lo()[i] == in_domain.lo()[this->perm[i]]);
  }
  m->num_dim = out_domain.get_dim();
  for (int i = 0; i < m->num_dim; i++)
    m->perm[i] = this->perm[i];
}

OpMeta* Transpose::init_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const Transpose* transpose = (const Transpose*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  Domain in_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain out_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());

  TransposeMeta* m = new TransposeMeta(handle);
  transpose->init_meta(m, in_domain, out_domain);
  m->profiling = transpose->profiling;
  return m;
}

void Transpose::init(const FFModel& ff)
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
  IndexLauncher launcher(TRANSPOSE_INIT_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Transpose)), argmap,
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

struct TransposeStrides
{
  int num_dim;
  int in_strides[MAX_TENSOR_DIM], out_strides[MAX_TENSOR_DIM], perm[MAX_TENSOR_DIM];
};

__global__
void transpose_simple_kernel(coord_t volume,
                             const float* in_ptr,
                             float* out_ptr,
                             const TransposeStrides info,
                             const float beta)
{
  CUDA_KERNEL_LOOP(o_idx, volume)
  {
    coord_t i_idx = 0;
    coord_t t = o_idx;
    for (int i = info.num_dim-1; i >= 0; i--) {
      coord_t ratio = t / info.out_strides[i];
      t -= ratio * info.out_strides[i];
      i_idx += ratio * info.in_strides[info.perm[i]];
    }
    out_ptr[o_idx] += out_ptr[o_idx] * beta + in_ptr[i_idx];
  }
}

/*static*/
void Transpose::forward_kernel(const TransposeMeta* m,
                               const float* input_ptr,
                               float* output_ptr,
                               Domain in_domain,
                               Domain out_domain,
                              cudaStream_t stream)
{
  TransposeStrides info;
  info.num_dim = out_domain.get_dim();
  assert(info.num_dim == m->num_dim);
  for (int i = 0; i < info.num_dim; i++) {
    int in_dim_size = (in_domain.hi()[i] - in_domain.lo()[i] + 1);
    int out_dim_size = (out_domain.hi()[i] - out_domain.lo()[i] + 1);
    info.in_strides[i] = (i == 0) ? 1 : info.in_strides[i-1] * in_dim_size;
    info.out_strides[i] = (i == 0) ? 1 : info.out_strides[i-1] * out_dim_size;
    info.perm[i] = m->perm[i];
  }
  transpose_simple_kernel<<<GET_BLOCKS(out_domain.get_volume()), CUDA_NUM_THREADS, 0, stream>>>(
      out_domain.get_volume(), input_ptr, output_ptr, info, 0.0f/*beta*/);
}

__host__
void Transpose::forward_task(const Task* task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Transpose* transpose = (const Transpose*) task->args;
  const TransposeMeta* m = *((TransposeMeta**) task->local_args);
  Domain in_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain out_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  for (int i = 0; i < out_domain.get_dim(); i++) {
    assert(out_domain.hi()[i] == in_domain.hi()[m->perm[i]]);
    assert(out_domain.lo()[i] == in_domain.lo()[m->perm[i]]);
  }
  const float* in_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* out_ptr = helperGetTensorPointerWO<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  forward_kernel(m, in_ptr, out_ptr, in_domain, out_domain, stream);
}

void Transpose::forward(const FFModel& ff)
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
  IndexLauncher launcher(TRANSPOSE_FWD_TASK_ID, task_is,
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

/*static*/
void Transpose::backward_kernel(const TransposeMeta* m,
                                float* input_grad_ptr,
                                const float* output_grad_ptr,
                                Domain in_grad_domain,
                                Domain out_grad_domain,
                                cudaStream_t stream)
{
  TransposeStrides info;
  info.num_dim = in_grad_domain.get_dim();
  assert(info.num_dim == m->num_dim);
  for (int i = 0; i < info.num_dim; i++) {
    int in_dim_size = (out_grad_domain.hi()[i] - out_grad_domain.lo()[i] + 1);
    int out_dim_size = (in_grad_domain.hi()[i] - in_grad_domain.lo()[i] + 1);
    info.in_strides[i] = (i == 0) ? 1 : info.in_strides[i-1] * in_dim_size;
    info.out_strides[i] = (i == 0) ? 1 : info.out_strides[i-1] * out_dim_size;
    info.perm[m->perm[i]] = i;
  }
  transpose_simple_kernel<<<GET_BLOCKS(in_grad_domain.get_volume()), CUDA_NUM_THREADS, 0, stream>>>(
      in_grad_domain.get_volume(), output_grad_ptr, input_grad_ptr, info, 1.0f/*beta*/);
}

__host__
void Transpose::backward_task(const Task* task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Transpose* transpose = (const Transpose*) task->args;
  const TransposeMeta* m = *((TransposeMeta**) task->local_args);
  Domain out_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain in_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  for (int i = 0; i < out_grad_domain.get_dim(); i++) {
    assert(out_grad_domain.hi()[i] == in_grad_domain.hi()[m->perm[i]]);
    assert(out_grad_domain.lo()[i] == in_grad_domain.lo()[m->perm[i]]);
  }
  const float* out_grad_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* in_grad_ptr = helperGetTensorPointerRW<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  backward_kernel(m, in_grad_ptr, out_grad_ptr, in_grad_domain, out_grad_domain, stream);
}

void Transpose::backward(const FFModel& ff)
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

  IndexLauncher launcher(TRANSPOSE_BWD_TASK_ID, task_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  // regions[0](I): output_grad
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
                      READ_ONLY, EXCLUSIVE, outputs[0].region_grad));
  launcher.add_field(0, FID_DATA);
  // regions[1](I/O): input_grad
  launcher.add_region_requirement(
    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
                      READ_WRITE, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

bool Transpose::measure_operator_cost(Simulator* sim,
                                      const ParallelConfig& pc,
                                      CostMetrics& cost_metrics)
{
  Tensor sub_input, sub_output;
  if (!outputs[0].get_output_sub_tensor(pc, sub_output, op_type)) {
    return false;
  }
  if (!inputs[0].get_input_sub_tensor(pc, sub_input, op_type)) {
    return false;
  }

  TransposeMeta *m = sim->transpose_meta;
  this->init_meta(m, sub_input.get_domain(), sub_output.get_domain());

  sim->free_all();
  float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert (input_ptr != NULL);
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_ptr != NULL);

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(m, input_ptr, output_ptr, sub_input.get_domain(), sub_output.get_domain(), stream);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *input_grad_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    assert (input_grad_ptr != NULL);
    float *output_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert (output_grad_ptr != NULL);
    backward = [&] {
      backward_kernel(m, input_grad_ptr, output_grad_ptr, sub_input.get_domain(), sub_output.get_domain(), stream);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure Transpose] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Measure Transpose] name(%s) forward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time);
  }

  return true;
}

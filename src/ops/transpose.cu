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
                          const std::vector<int>& perm)
{
  Transpose* transpose = new Transpose(*this, input, perm);
  layers.push_back(transpose);
  return transpose->outputs[0];
}

Transpose::Transpose(FFModel& model,
                     const Tensor& input,
                     const std::vector<int>& _perm)
: Op(model, OP_TRANSPOSE, "Transpose_", input)
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

Tensor Transpose::init_inout(FFModel& model,
                             const Tensor& input)
{
  inputs[0] = input;
  create_output_and_partition(model);
  return outputs[0];
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
      // Unsupported dim for ElementWiseUnary operator
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

OpMeta* Transpose::init_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)
{
  return NULL;
}

void Transpose::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(TRANSPOSE_INIT_TASK_ID, task_is,
                         TaskArgument(this, sizeof(ElementBinary)), argmap,
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

__host__
void Transpose::forward_task(const Task* task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const Transpose* transpose = (const Transpose*) task->args;
  Domain in_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain out_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  for (int i = 0; i < out_domain.get_dim(); i++) {
    assert(out_domain.hi()[i] == in_domain.hi()[transpose->perm[i]]);
    assert(out_domain.lo()[i] == in_domain.lo()[transpose->perm[i]]);
  }
  const float* in_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* out_ptr = helperGetTensorPointerWO<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  TransposeStrides info;
  info.num_dim = out_domain.get_dim();
  for (int i = 0; i < info.num_dim; i++) {
    int in_dim_size = (in_domain.hi()[i] - in_domain.lo()[i] + 1);
    int out_dim_size = (out_domain.hi()[i] - out_domain.lo()[i] + 1);
    info.in_strides[i] = (i == 0) ? 1 : info.in_strides[i-1] * in_dim_size;
    info.out_strides[i] = (i == 0) ? 1 : info.out_strides[i-1] * out_dim_size;
    info.perm[i] = transpose->perm[i];
  }
  transpose_simple_kernel<<<GET_BLOCKS(out_domain.get_volume()), CUDA_NUM_THREADS>>>(
      out_domain.get_volume(), in_ptr, out_ptr, info, 0.0f/*beta*/);
}

void Transpose::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(TRANSPOSE_FWD_TASK_ID, task_is,
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
  runtime->execute_index_space(ctx, launcher);
}

__host__
void Transpose::backward_task(const Task* task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const Transpose* transpose = (const Transpose*) task->args;
  Domain out_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain in_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  for (int i = 0; i < out_grad_domain.get_dim(); i++) {
    assert(out_grad_domain.hi()[i] == in_grad_domain.hi()[transpose->perm[i]]);
    assert(out_grad_domain.lo()[i] == in_grad_domain.lo()[transpose->perm[i]]);
  }
  const float* out_grad_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* in_grad_ptr = helperGetTensorPointerRW<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  TransposeStrides info;
  info.num_dim = in_grad_domain.get_dim();
  for (int i = 0; i < info.num_dim; i++) {
    int in_dim_size = (out_grad_domain.hi()[i] - out_grad_domain.lo()[i] + 1);
    int out_dim_size = (in_grad_domain.hi()[i] - in_grad_domain.lo()[i] + 1);
    info.in_strides[i] = (i == 0) ? 1 : info.in_strides[i-1] * in_dim_size;
    info.out_strides[i] = (i == 0) ? 1 : info.out_strides[i-1] * out_dim_size;
    info.perm[transpose->perm[i]] = i;
  }
  transpose_simple_kernel<<<GET_BLOCKS(in_grad_domain.get_volume()), CUDA_NUM_THREADS>>>(
      in_grad_domain.get_volume(), out_grad_ptr, in_grad_ptr, info, 1.0f/*beta*/);
}

void Transpose::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(TRANSPOSE_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Transpose)), argmap,
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

bool Transpose::measure_compute_time(Simulator* sim,
                                     const ParallelConfig& pc,
                                     float& forward_time,
                                     float& backward_time)
{
  //TODO: implement measure_forward
  return false;
}

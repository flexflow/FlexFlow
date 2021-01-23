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

void FFModel::split(const Tensor& input,
                    Tensor* outputs,
                    const std::vector<int>& splits,
                    int axis,
                    const char* name)
{
  Split* split = new Split(*this, input, splits, axis, name);
  layers.push_back(split);
  for (size_t i = 0; i < splits.size(); i++)
    outputs[i] = split->outputs[i];
}

Split::Split(FFModel& model,
             const Tensor& input,
             const std::vector<int>& splits,
             int _axis,
             const char* name)
: Op(model, OP_SPLIT, name, input)
{
  numOutputs = splits.size();
  // Use the Legion dim ordering
  axis = input.numDim - 1 - _axis;
  assert(axis >= 0);
  numWeights = 0;
  int split_size = 0;
  for (int i = 0; i < numOutputs; i++) {
    split_size += splits[i];
    outputs[i].numDim = input.numDim;
    for (int j = 0; j < input.numDim; j++)
      outputs[i].adim[j] = input.adim[j];
    outputs[i].adim[axis] = splits[i];
  }
  // Check split sizes
  assert(split_size == input.adim[axis]);
}

void Split::create_weights(FFModel& model)
{
  // Do nothing
}

void Split::create_output_and_partition(FFModel& model)
{
  // Retrive the task indexspace
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
      // Unsupported dim for Split operator
      assert(false);
    }
  }
}

template<int NDIM>
void Split::create_output_and_partition_with_dim(FFModel& model)
{
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  // cannot parallelize along the axis dim
  assert(part_rect.hi[axis] == part_rect.lo[axis]);
  for (int i = 0; i < numOutputs; i++) {
    int dims[NDIM];
    for (int j = 0; j < NDIM; j++)
      dims[j] = outputs[i].adim[NDIM-1-j];
    outputs[i] = model.create_tensor<NDIM>(dims, DT_FLOAT, this);
    outputs[i].owner_op = this;
    outputs[i].owner_idx = i;
  }
  Rect<NDIM> input_rect = runtime->get_index_partition_color_space(
      ctx, inputs[0].part.get_index_partition());
  if (input_rect == part_rect) {
    input_lps[0] = inputs[0].part;
    input_grad_lps[0] = inputs[0].part_grad;
  } else {
    model.create_disjoint_partition<NDIM>(
        inputs[0], IndexSpaceT<NDIM>(task_is), input_lps[0], input_grad_lps[0]);
  }
}

__host__
OpMeta* Split::init_task(const Task* task,
                         const std::vector<PhysicalRegion>& regions,
                         Context ctx, Runtime* runtime)
{
  return NULL;
}

void Split::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(SPLIT_INIT_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Split)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numOutputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(outputs[i].part, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, outputs[i].region));
    launcher.add_field(i+1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

void calc_block_size(coord_t& num_blks,
                     coord_t& blk_size,
                     const Domain& domain,
                     int axis)
{
  num_blks = 1;
  blk_size = 1;
  for (int d = 0; d < domain.get_dim(); d++) {
    if (d <= axis)
      blk_size *= (domain.hi()[d] - domain.lo()[d] + 1);
    else
      num_blks *= (domain.hi()[d] - domain.lo()[d] + 1);
  }
}

void Split::forward_kernel(float **out_ptrs,
                           float const *in_ptr,
                           coord_t const *out_blk_sizes,
                           coord_t in_blk_size,
                           coord_t num_blks,
                           int numOutputs)
{
  for (int i = 0; i < numOutputs; i++) {
    copy_with_stride<<<GET_BLOCKS(out_blk_sizes[i]*num_blks), CUDA_NUM_THREADS>>>(
        out_ptrs[i], in_ptr, num_blks, out_blk_sizes[i], in_blk_size);
    in_ptr += out_blk_sizes[i];
  }
}

void Split::forward_task(const Task *task,
                         const std::vector<PhysicalRegion>& regions,
                         Context ctx, Runtime *runtime)
{
  const Split* split = (Split*) task->args;
  assert(regions.size() == split->numOutputs + 1);
  assert(task->regions.size() == split->numOutputs + 1);
  Domain in_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  float* out_ptr[MAX_NUM_OUTPUTS];
  size_t total_volume = 0;
  const float* in_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  coord_t num_blks, in_blk_size, out_blk_size[MAX_NUM_OUTPUTS];
  calc_block_size(num_blks, in_blk_size, in_domain, split->axis);
  for (int i = 0; i < split->numOutputs; i++) {
    Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+1].region.get_index_space());
    out_ptr[i] = helperGetTensorPointerWO<float>(
      regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime);
    coord_t out_num_blks;
    calc_block_size(out_num_blks, out_blk_size[i], out_domain, split->axis);
    assert(out_num_blks == num_blks);
    for (int j = 0; j < out_domain.get_dim(); j++)
      if (j != split->axis) {
        assert(out_domain.hi()[j] == in_domain.hi()[j]);
        assert(out_domain.lo()[j] == in_domain.lo()[j]);
      }
    total_volume += out_domain.get_volume();
  }
  assert(total_volume == in_domain.get_volume());
  forward_kernel(out_ptr, in_ptr, out_blk_size, in_blk_size, num_blks, split->numOutputs);
}

void Split::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(SPLIT_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Split)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numOutputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(outputs[i].part, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, outputs[i].region));
    launcher.add_field(i+1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

void Split::backward_task(const Task *task,
                          const std::vector<PhysicalRegion>& regions,
                          Context ctx, Runtime *runtime)
{
  const Split* split = (Split*) task->args;
  assert(regions.size() == split->numOutputs + 1);
  assert(task->regions.size() == split->numOutputs + 1);
  Domain in_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  const float* out_grad_ptr[MAX_NUM_OUTPUTS];
  size_t total_volume = 0;
  float* in_grad_ptr = helperGetTensorPointerRW<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  coord_t num_blks, in_blk_size, out_blk_size[MAX_NUM_OUTPUTS];
  calc_block_size(num_blks, in_blk_size, in_grad_domain, split->axis);
  for (int i = 0; i < split->numOutputs; i++) {
    Domain out_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+1].region.get_index_space());
    out_grad_ptr[i] = helperGetTensorPointerRO<float>(
      regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime);
    coord_t out_num_blks;
    calc_block_size(out_num_blks, out_blk_size[i], out_grad_domain, split->axis);
    assert(out_num_blks == num_blks);
    for (int j = 0; j < out_grad_domain.get_dim(); j++)
      if (j != split->axis) {
        assert(out_grad_domain.hi()[j] == in_grad_domain.hi()[j]);
        assert(out_grad_domain.lo()[j] == in_grad_domain.lo()[j]);
      }
    total_volume += out_grad_domain.get_volume();
  }
  assert(total_volume == in_grad_domain.get_volume());
  for (int i = 0; i < split->numOutputs; i++) {
    add_with_stride<<<GET_BLOCKS(out_blk_size[i]*num_blks), CUDA_NUM_THREADS>>>(
        in_grad_ptr, out_grad_ptr[i], num_blks, in_blk_size, out_blk_size[i]);
    in_grad_ptr += out_blk_size[i];
  }
  //checkCUDA(cudaDeviceSynchronize());
}

void Split::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(SPLIT_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Split)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numOutputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(outputs[i].part_grad, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, outputs[i].region_grad));
    launcher.add_field(i+1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

bool Split::measure_compute_time(Simulator* sim,
                                 const ParallelConfig& pc,
                                 float& forward_time,
                                 float& backward_time)
{
  //TODO: implement measure_forward
  forward_time = 0.0f;
  backward_time = 0.0f;
  return false;
}

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

Tensor FFModel::reverse(const Tensor& input,
                        int axis,
                        const char* name)
{
  Reverse* reverse = new Reverse(*this, input, axis, name);
  layers.push_back(reverse);
  return reverse->outputs[0];
}

Reverse::Reverse(FFModel& model,
                 const Tensor& input,
                 int _axis,
                 const char* name)
: Op(model, OP_REVERSE, name, input), axis(_axis)
{
  outputs[0].numDim = input.numDim;
  for (int i = 0; i < input.numDim; i++)
    outputs[0].adim[i] = input.adim[i];
  numInputs = 1;
  numWeights = 0;
}

void Reverse::create_weights(FFModel& model)
{
  // Do nothing since no weights
}

void Reverse::create_output_and_partition(FFModel& model)
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
      // Unsupported dim for Reverse operator
      assert(false);
    }
  }
}

template<int NDIM>
void Reverse::create_output_and_partition_with_dim(FFModel& model)
{
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  // the degree of parallelism along the reversed dimension must be 1
  assert(part_rect.hi[NDIM-1-axis] == part_rect.lo[NDIM-1-axis]);
  int dims[NDIM];
  for (int i = 0; i < NDIM; i++)
    dims[i] = outputs[0].adim[NDIM-1-i];
  outputs[0] = model.create_tensor<NDIM>(dims, DT_FLOAT, this);
  outputs[0].owner_op = this;
  outputs[0].owner_idx = 0;
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
OpMeta* Reverse::init_task(const Task* task,
                           const std::vector<PhysicalRegion>& regions,
                           Context ctx, Runtime* runtime)
{
  return NULL;
}

void Reverse::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(REVERSE_INIT_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Reverse)), argmap,
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

__global__
void reverse_forward_kernel(const float* in_ptr,
                            float* out_ptr,
                            coord_t num_out_blks,
                            coord_t reverse_dim_size,
                            coord_t in_blk_size)
{
  CUDA_KERNEL_LOOP(i, num_out_blks * reverse_dim_size * in_blk_size)
  {
    coord_t blk_idx = i / (reverse_dim_size * in_blk_size);
    i = i - blk_idx * (reverse_dim_size * in_blk_size);
    coord_t reverse_dim_idx = i / in_blk_size;
    i = i - reverse_dim_idx * in_blk_size;
    coord_t in_idx = blk_idx * (reverse_dim_size * in_blk_size)
                   + (reverse_dim_size - 1 - reverse_dim_idx) * in_blk_size + i;
    out_ptr[i] = in_ptr[in_idx];
  }
}

void Reverse::forward_kernel(float const *in_ptr,
                             float *out_ptr,
                             coord_t num_out_blks,
                             coord_t reverse_dim_size,
                             coord_t in_blk_size,
                             coord_t output_size,
                             cudaStream_t stream)
{
  reverse_forward_kernel<<<GET_BLOCKS(output_size), CUDA_NUM_THREADS, 0, stream>>>(
      in_ptr, out_ptr, num_out_blks, reverse_dim_size, in_blk_size);
}


__host__
void Reverse::forward_task(const Task* task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const Reverse* reverse = (const Reverse*) task->args;
  Domain in_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain out_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  assert(out_domain == in_domain);
  const float* in_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* out_ptr = helperGetTensorPointerWO<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  int axis = in_domain.get_dim() - reverse->axis - 1;
  coord_t in_blk_size = 1, reverse_dim_size = 1, num_out_blks = 1;
  for (int i = 0; i < out_domain.get_dim(); i++) {
    if (i < axis)
      in_blk_size *= out_domain.hi()[i] - out_domain.lo()[i] + 1;
    else if (i == axis)
      reverse_dim_size = out_domain.hi()[i] - out_domain.lo()[i] + 1;
    else
      num_out_blks *= out_domain.hi()[i] - out_domain.lo()[i] + 1;
  }
  int output_size = out_domain.get_volume();

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  forward_kernel(in_ptr, out_ptr, num_out_blks, reverse_dim_size, in_blk_size, output_size, stream);
}

void Reverse::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(REVERSE_FWD_TASK_ID, task_is,
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

void Reverse::backward_kernel(float const *out_grad_ptr,
                              float *in_grad_ptr,
                              coord_t num_out_blks,
                              coord_t reverse_dim_size,
                              coord_t in_blk_size,
                              coord_t input_size,
                              cudaStream_t stream)
{
  reverse_forward_kernel<<<GET_BLOCKS(input_size), CUDA_NUM_THREADS, 0, stream>>>(
      out_grad_ptr, in_grad_ptr, num_out_blks, reverse_dim_size, in_blk_size);
}

__host__
void Reverse::backward_task(const Task* task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const Reverse* reverse = (const Reverse*) task->args;
  Domain out_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain in_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  assert(out_grad_domain == in_grad_domain);
  const float* out_grad_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* in_grad_ptr = helperGetTensorPointerRW<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  // We reuse the forward kernel for backward tasks
  int axis = in_grad_domain.get_dim() - reverse->axis - 1;
  coord_t in_blk_size = 1, reverse_dim_size = 1, num_out_blks = 1;
  for (int i = 0; i < in_grad_domain.get_dim(); i++) {
    if (i < axis)
      in_blk_size *= in_grad_domain.hi()[i] - in_grad_domain.lo()[i] + 1;
    else if (i == axis)
      reverse_dim_size = in_grad_domain.hi()[i] - in_grad_domain.lo()[i] + 1;
    else
      num_out_blks *= in_grad_domain.hi()[i] - in_grad_domain.lo()[i] + 1;
  }

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  backward_kernel(out_grad_ptr, in_grad_ptr, num_out_blks, reverse_dim_size, in_blk_size, in_grad_domain.get_volume(), stream);
}

void Reverse::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(REVERSE_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Linear)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  // regions[0](I): output_grad
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
                      READ_ONLY, EXCLUSIVE, outputs[0].region_grad));
  launcher.add_field(0, FID_DATA);
  // regions[1](I/O): input0_grad
  launcher.add_region_requirement(
    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
                      READ_WRITE, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

bool Reverse::measure_operator_cost(Simulator* sim,
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

  sim->free_all();
  float *input_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert (input_ptr != NULL);
  float *output_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_ptr != NULL);

  coord_t in_blk_size = 1, reverse_dim_size = 1, num_out_blks = 1;
  for (int i = 0; i < sub_output.numDim; i++) {
    if (i < axis) {
      in_blk_size *= sub_output.adim[i];
    } else if (i == axis) {
      reverse_dim_size = sub_output.adim[i];
    } else {
      num_out_blks *= sub_output.adim[i];
    }
  }

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  std::function<void()> forward, backward;
  forward = [&] {
     forward_kernel(input_ptr, output_ptr, num_out_blks, reverse_dim_size, in_blk_size, sub_output.get_volume(), stream);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *input_grad_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    assert (input_grad_ptr != NULL);
    float *output_grad_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert (output_grad_ptr != NULL);
    backward = [&] {
      backward_kernel(output_grad_ptr, input_grad_ptr, num_out_blks, reverse_dim_size, in_blk_size, sub_input.get_volume(), stream);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure Reverse] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Measure Reverse] name(%s) forward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time);
  }

  return true;
}

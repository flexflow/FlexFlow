/* Copyright 2020 Stanford
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

Tensor FFModel::exp(std::string name,
                    const Tensor& x)
{
  ElementUnary *ele = new ElementUnary(*this, ElementUnary::OP_EXP, name, x);
  ele->add_to_model(*this);
  return ele->outputs[0];
}

ElementUnary* FFModel::exp(std::string name)
{
  ElementUnary* ele = new ElementUnary(*this, ElementUnary::OP_EXP, name);
  return ele;
}

ElementUnary::ElementUnary(FFModel& model,
                           ElementUnary::OpType _op_type,
                           const std::string& pcname,
                           const Tensor& x)
: Op(pcname, x), op_type(_op_type)
{
  int dim = x.numDim;
  switch (dim) {
    case 1:
    {
      task_is = model.get_or_create_task_is(1, name);
      create_output_and_partition<1>(model);
      break;
    }
    case 2:
    {
      task_is = model.get_or_create_task_is(2, name);
      create_output_and_partition<2>(model);
      break;
    }
    case 3:
    {
      task_is = model.get_or_create_task_is(3, name);
      create_output_and_partition<3>(model);
      break;
    }
    case 4:
    {
      task_is = model.get_or_create_task_is(4, name);
      create_output_and_partition<4>(model);
      break;
    }
    default:
    {
      // Unsupported dim for ElementBinarywise operator
      assert(false);
    }
  }
}

ElementUnary::ElementUnary(FFModel& model,
                           ElementUnary::OpType _op_type,
                           const std::string& pcname)
: Op(pcname), op_type(_op_type)
{}

Tensor ElementUnary::init_inout(FFModel& model,
                                const Tensor& input)
{
  add_to_model(model);
  inputs[0] = input;
  int dim = input.numDim;
  switch (dim) {
    case 1:
    {
      task_is = model.get_or_create_task_is(1, name);
      create_output_and_partition<1>(model);
      break;
    }
    case 2:
    {
      task_is = model.get_or_create_task_is(2, name);
      create_output_and_partition<2>(model);
      break;
    }
    case 3:
    {
      task_is = model.get_or_create_task_is(3, name);
      create_output_and_partition<3>(model);
      break;
    }
    case 4:
    {
      task_is = model.get_or_create_task_is(4, name);
      create_output_and_partition<4>(model);
      break;
    }
    default:
    {
      // Unsupported dim for ElementWiseUnary operator
      assert(false);
    }
  }
  return outputs[0];
}

void ElementUnary::add_to_model(FFModel& model)
{
  model.layers.push_back(this);
}

template<int NDIM>
void ElementUnary::create_output_and_partition(FFModel& model)
{
  // Retrive the task indexspace for the op
  task_is = IndexSpaceT<NDIM>(model.get_or_create_task_is(NDIM, name));
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int dims[NDIM];
  for (int i = 0; i < NDIM; i++)
    dims[i] = inputs[0].adim[NDIM-1-i];
  outputs[0] = model.create_tensor<NDIM>(dims, IndexSpaceT<NDIM>(task_is), DT_FLOAT);
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

void ElementUnary::init(const FFModel& ff)
{
}

__global__
void elewise_unary_forward_kernel(coord_t volume,
                                  const float alpha,
                                  const float beta,
                                  ElementUnary::OpType type,
                                  const float* in,
                                  float* out)
{
  CUDA_KERNEL_LOOP(i, volume)
  {
    switch (type) {
      case ElementUnary::OP_EXP:
      {
        out[i] = alpha * exp(in[i]) + beta * out[i];
        break;
      }
      default:
        assert(false);
    }
  }
}

/*
  regions[0](I): input
  regions[1](O): output
*/
__host__
void ElementUnary::forward_task(const Task* task,
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx, Runtime* runtime)
{
  float alpha = 1.0f, beta = 0.0f;
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const ElementUnary* ele = (const ElementUnary*) task->args;
  Domain input_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  assert(output_domain == input_domain);

  const float* input_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* output_ptr = helperGetTensorPointerWO<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  elewise_unary_forward_kernel<<<GET_BLOCKS(output_domain.get_volume()), CUDA_NUM_THREADS>>>(
  output_domain.get_volume(), alpha, beta, ele->op_type, input_ptr, output_ptr);
}

void ElementUnary::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(ELEMENTUNARY_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(ElementUnary)), argmap,
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
void elewise_unary_backward_kernel(coord_t volume,
                                   const float alpha,
                                   const float beta,
                                   ElementUnary::OpType type,
                                   const float* output_grad,
                                   const float* input,
                                   float* input_grad)
{
  CUDA_KERNEL_LOOP(i, volume)
  {
    switch (type) {
      case ElementUnary::OP_EXP:
      {
        //TODO: change to use output instead of recomputing
        input_grad[i] = alpha * output_grad[i] * exp(input[i]) + beta * input_grad[i];
        break;
      }
      default:
        assert(false);
    }
  }
}

/*
  regions[0](I): output_grad
  regions[1](I): input
  regions[2](I/O): input_grad
*/
__host__
void ElementUnary::backward_task(const Task* task,
                                 const std::vector<PhysicalRegion> &regions,
                                 Context ctx, Runtime* runtime)
{
  float alpha = 1.0f;
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  const ElementUnary* ele = (const ElementUnary*) task->args;
  Domain output_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain input_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  Domain input_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[2].region.get_index_space());
  assert(output_grad_domain == input_domain);
  assert(output_grad_domain == input_grad_domain);

  const float* output_grad_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  const float* input_ptr = helperGetTensorPointerRO<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  float* input_grad_ptr = helperGetTensorPointerRW<float>(
    regions[2], task->regions[2], FID_DATA, ctx, runtime);
  elewise_unary_backward_kernel<<<GET_BLOCKS(input_domain.get_volume()), CUDA_NUM_THREADS>>>(
    input_domain.get_volume(), alpha, alpha, ele->op_type, output_grad_ptr, input_ptr, input_grad_ptr);
}

void ElementUnary::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(ELEMENTUNARY_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(ElementUnary)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  // regions[0](I): output_grad
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
                      READ_ONLY, EXCLUSIVE, outputs[0].region_grad));
  launcher.add_field(0, FID_DATA);
  // regions[1](I): input
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
                      READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(1, FID_DATA);
  // regions[4](I/O): input1_grad
  launcher.add_region_requirement(
    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
                      READ_WRITE, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}



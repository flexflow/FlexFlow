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

#include "ops/element_unary.h"
#include "hash_utils.h"
#include "cuda_helper.h"

using namespace Legion;

Tensor FFModel::unary(OperatorType op,
                      const Tensor x,
                      bool inplace,
                      const char *name,
		      float scalar)
{
  ElementUnary *ele = new ElementUnary(*this, op, x, inplace, name, scalar);
  layers.push_back(ele);
  return ele->outputs[0];
}

size_t ElementUnary::get_params_hash() const {
  size_t hash = this->inputs[0]->get_owner_independent_hash();
  hash_combine(hash, this->op_type);
  hash_combine(hash, this->inplace);
  if (this->op_type == OP_SCALAR_MULTIPLY) {
    hash_combine(hash, this->scalar);
  }

  return hash;
}

Node FFModel::get_or_create_element_unary_node(const Tensor input,
                                               OperatorType op,
                                               bool inplace,
                                               float scalar)
{
  if (input->dims[input->num_dims-1].degree != 1) {
    return Node::INVALID_NODE;
  }

  size_t hash = input->get_owner_independent_hash();
  hash_combine(hash, op);
  hash_combine(hash, inplace);
  if (op == OP_SCALAR_MULTIPLY) {
    hash_combine(hash, scalar);
  }

  ElementUnary *unary;
  const auto &it = this->cached_element_unary_ops.find(hash);
  if (it != cached_element_unary_ops.end()) { 
    unary = it->second;
  } else {
    unary = new ElementUnary(*this, op, input, inplace, NULL, scalar);
    cached_element_unary_ops[hash] = unary;
  }

  return this->new_node(unary);
}

Tensor FFModel::exp(const Tensor x,
                    const char *name)
{
  return this->unary(OP_EXP, x, false/*inplace*/, name);
}

Tensor FFModel::scalar_multiply(const Tensor x, const float scalar, bool inplace, const char *name)
{
  return this->unary(OP_SCALAR_MULTIPLY, x, inplace, name, scalar);
}

Tensor FFModel::relu(const Tensor x, bool inplace, const char *name)
{
  return this->unary(OP_RELU, x, inplace, name);
}

Tensor FFModel::sigmoid(const Tensor x, const char *name)
{
  return this->unary(OP_SIGMOID, x, false/*inplace*/, name);
}

Tensor FFModel::tanh(const Tensor x, const char *name)
{
  return this->unary(OP_TANH, x, false/*inplace*/, name);
}

Tensor FFModel::identity(const Tensor x, const char *name)
{
  return this->unary(OP_IDENTITY, x, false/*inplace*/, name);
}

Tensor FFModel::gelu(const Tensor x, const char *name)
{
  return this->unary(OP_GELU, x, false/*inplace*/, name);
}

Tensor FFModel::elu(const Tensor x, bool inplace, const char *name)
{
  // Currently assume inplace is false
  assert(!inplace);
  return this->unary(OP_ELU, x, inplace, name);
}

ElementUnary::ElementUnary(FFModel& model,
                           OperatorType _op_type,
                           const Tensor x,
                           bool _inplace,
                           const char* name,
                           float _scalar)
: Op(model, _op_type, name, 1/*inputs*/, 0/*weights*/, 1/*outputs*/, x), inplace(_inplace), scalar(_scalar)
{
  numOutputs = 1;
  int numdim = x->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = x->dims[i];
  }
  outputs[0] = model.create_tensor_legion_ordering(numdim, dims, x->data_type, this);
}

bool ElementUnary::can_inplace_output(void)
{
  return true;
}

bool ElementUnary::has_inplace_output(void)
{
  return inplace;
}

void ElementUnary::do_inplace_output(void)
{
  inplace = true;
}

bool ElementUnary::use_cudnn(OperatorType type)
{
  if (type == OP_RELU)
    return true;
  if (type == OP_SIGMOID)
    return true;
  if (type == OP_TANH)
    return true;
  if (type == OP_ELU)
    return true;
  return false;
}

OpMeta* ElementUnary::init_task(const Task *task,
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx, Runtime *runtime)
{
  ElementUnary* eu = (ElementUnary*) task->args;
  FFHandler handle = *((FFHandler*) task->local_args);
  ElementUnaryMeta* m = new ElementUnaryMeta(handle);
  m->op_type = eu->op_type;
  m->profiling = eu->profiling;
  m->inplace = eu->inplace;
  m->scalar = eu->scalar;
  if (m->inplace) {
    assert(regions.size() == 1);
    assert(task->regions.size() == 1);
  } else {
    assert(regions.size() == 2);
    assert(task->regions.size() == 2);
  }

  if (use_cudnn(m->op_type))
  {
    cudnnActivationMode_t mode;
    switch (m->op_type) {
      case OP_SIGMOID:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      case OP_RELU:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case OP_TANH:
        mode = CUDNN_ACTIVATION_TANH;
        break;
      case OP_ELU:
        mode = CUDNN_ACTIVATION_ELU;
        break;
      default:
        assert(false);
    }
    checkCUDNN(cudnnSetActivationDescriptor(m->actiDesc, mode,
                                            CUDNN_PROPAGATE_NAN, 0.0));
    Domain input_domain = runtime->get_index_space_domain(
        ctx, task->regions[0].region.get_index_space());
    checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->inputTensor, input_domain));
    checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->outputTensor, input_domain));
  }
  return m;
}

void ElementUnary::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher init_launcher(ELEMENTUNARY_INIT_TASK_ID, parallel_is,
                              TaskArgument(this, sizeof(ElementUnary)), argmap,
                              Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                              outputs[0]->machine_view.hash());
  init_launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  init_launcher.add_field(0, FID_DATA);
  assert (!inplace);
  if (!inplace) {
    init_launcher.add_region_requirement(
        RegionRequirement(outputs[0]->part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
    init_launcher.add_field(1, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

__global__
void elewise_unary_forward_kernel(coord_t volume,
                                  const float alpha,
                                  const float beta,
                                  const float scalar,
				  OperatorType type,
                                  const float* in,
                                  float* out)
{
  CUDA_KERNEL_LOOP(i, volume)
  {
    switch (type) {
      case OP_EXP:
      {
        out[i] = alpha * exp(in[i]) + beta * out[i];
        break;
      }
      case OP_IDENTITY:
      {
	out[i] = in[i];
	break;
      }
      case OP_SCALAR_MULTIPLY:
      {
	out[i] = in[i] * scalar;
	break;
      }
      case OP_GELU:
      {
	out[i] = in[i] * 0.5 * erfc(-in[i]*M_SQRT1_2);
	break;
      }
      default:
        assert(false);
    }
  }
}

/*static*/
void ElementUnary::forward_kernel(const ElementUnaryMeta* m,
                                  const float* input_ptr,
                                  float* output_ptr,
                                  size_t num_elements)
{
  float alpha = 1.0f, beta = 0.0f;
  if (use_cudnn(m->op_type)) {
    checkCUDNN(cudnnActivationForward(m->handle.dnn, m->actiDesc,
        &alpha, m->inputTensor, input_ptr,
        &beta, m->outputTensor, output_ptr));
  } else {
    elewise_unary_forward_kernel<<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS>>>(
        num_elements, alpha, beta,m->scalar, m->op_type, input_ptr, output_ptr);
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
  //const ElementUnary* ele = (const ElementUnary*) task->args;
  const ElementUnaryMeta* m = *((ElementUnaryMeta**) task->local_args);
  Domain input_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  const float* input_ptr = NULL;
  float* output_ptr = NULL;
  if (m->inplace) {
    assert(regions.size() == 1);
    assert(task->regions.size() == 1);
    output_ptr = helperGetTensorPointerRW<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
    input_ptr = output_ptr;
  } else {
    assert(regions.size() == 2);
    assert(task->regions.size() == 2);
    Domain output_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
    assert(output_domain == input_domain);
    input_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
    output_ptr = helperGetTensorPointerWO<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  }
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  forward_kernel(m, input_ptr, output_ptr, input_domain.get_volume());
}

void ElementUnary::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(ELEMENTUNARY_FWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  if (inplace) {
    assert(outputs[0]->part == inputs[0]->part);
    assert(outputs[0]->region == inputs[0]->region);
    launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, outputs[0]->region));
    launcher.add_field(0, FID_DATA);
  } else {
    launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[0]->region));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
         WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
    launcher.add_field(1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

__global__
void elewise_unary_backward_kernel(coord_t volume,
                                   const float alpha,
                                   const float beta,
				   const float scalar,
                                   OperatorType type,
                                   const float* output_grad,
                                   const float* input,
                                   float* input_grad)
{
  CUDA_KERNEL_LOOP(i, volume)
  {
    switch (type) {
      case OP_EXP:
      {
        //TODO: change to use output instead of recomputing
        input_grad[i] = alpha * output_grad[i] * exp(input[i]) + beta * input_grad[i];
        break;
      }
      case OP_IDENTITY:
      {
	input_grad[i] = output_grad[i];
	break;
      } 
      case OP_SCALAR_MULTIPLY:
      {
	input_grad[i] = output_grad[i]*scalar;
	break;
      }
      case OP_GELU:
      {
	input_grad[i] = output_grad[i]*(0.5 * erfc(-input[i]*M_SQRT1_2)-0.5*M_SQRT1_2*input[i]*exp(-input[i]*input[i]*0.5));
	break;
      }
      default:
        assert(false);
    }
  }
}

/*static*/
void ElementUnary::backward_kernel(const ElementUnaryMeta* m,
                                   const float* input_ptr,
                                   float* input_grad_ptr,
                                   const float* output_ptr,
                                   const float* output_grad_ptr,
                                   size_t num_elements)
{
  float alpha = 1.0f;
  if (use_cudnn(m->op_type)) {
    checkCUDNN(cudnnActivationBackward(m->handle.dnn, m->actiDesc,
        &alpha, m->outputTensor, output_ptr, m->outputTensor, output_grad_ptr,
        m->inputTensor, input_ptr, &alpha, m->inputTensor, input_grad_ptr));
  } else {
    elewise_unary_backward_kernel<<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS>>>(
        num_elements, alpha, alpha, m->scalar, m->op_type, output_grad_ptr, input_ptr, input_grad_ptr);
  }
}

/*
  regions[0](I): input
  regions[1](I/O): input_grad
  regions[2](I): output
  regions[3](I): output_grad
*/
__host__
void ElementUnary::backward_task(const Task* task,
                                 const std::vector<PhysicalRegion> &regions,
                                 Context ctx, Runtime* runtime)
{
  //const ElementUnary* ele = (const ElementUnary*) task->args;
  const ElementUnaryMeta* m = *((ElementUnaryMeta**) task->local_args);
  const float* input_ptr = NULL, *output_ptr = NULL, *output_grad_ptr = NULL;
  float* input_grad_ptr = NULL;
  Domain input_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  if (m->inplace) {
    assert(regions.size() == 2);
    assert(task->regions.size() == 2);
    Domain input_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
    assert(input_grad_domain == input_domain);
    input_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
    input_grad_ptr = helperGetTensorPointerRW<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
    output_ptr = input_ptr;
    output_grad_ptr = input_grad_ptr;
  } else {
    assert(regions.size() == 4);
    assert(task->regions.size() == 4);
    Domain input_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
    Domain output_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
    Domain output_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[3].region.get_index_space());
    assert(output_grad_domain == input_domain);
    assert(output_grad_domain == output_domain);
    assert(output_grad_domain == input_grad_domain);
    input_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
    input_grad_ptr = helperGetTensorPointerRW<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
    output_ptr = helperGetTensorPointerRO<float>(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
    output_grad_ptr = helperGetTensorPointerRO<float>(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);
  }
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  backward_kernel(m, input_ptr, input_grad_ptr, output_ptr, output_grad_ptr, input_domain.get_volume());
}

void ElementUnary::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(ELEMENTUNARY_BWD_TASK_ID, parallel_is,
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      outputs[0]->machine_view.hash());
  if (inplace) {
    assert(inputs[0]->part == outputs[0]->part);
    assert(inputs[0]->part_grad == outputs[0]->part_grad);
    // regions[2](I): output_grad
    launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, outputs[0]->region));
    launcher.add_field(0, FID_DATA);
    // regions[3](I): output_grad
    launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, outputs[0]->region_grad));
    launcher.add_field(1, FID_DATA);
  } else {
    // regions[0](I): input
    launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[0]->region));
    launcher.add_field(0, FID_DATA);
    // regions[1](I/O): input_grad
    launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
    launcher.add_field(1, FID_DATA);
    // regions[2](I): output_grad
    launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, outputs[0]->region));
    launcher.add_field(2, FID_DATA);
    // regions[3](I): output_grad
    launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
    launcher.add_field(3, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

ElementUnaryMeta::ElementUnaryMeta(FFHandler handler)
: OpMeta(handler)
{
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
}

bool ElementUnary::measure_operator_cost(Simulator* sim,
                                         const ParallelConfig& pc,
                                         CostMetrics& cost_metrics) const
{
  TensorBase sub_output, sub_input;
  if (!outputs[0]->get_output_sub_tensor(pc, sub_output, op_type))
    return false;
  if (!inputs[0]->get_input_sub_tensor(pc, sub_input, op_type))
    return false;
  ElementUnaryMeta* m = sim->ele_unary_meta;
  m->op_type = op_type;
  if (use_cudnn(m->op_type))
  {
    cudnnActivationMode_t mode;
    switch (op_type) {
      case OP_SIGMOID:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      case OP_RELU:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case OP_TANH:
        mode = CUDNN_ACTIVATION_TANH;
        break;
      case OP_ELU:
        mode = CUDNN_ACTIVATION_ELU;
        break;
      default:
        assert(false);
    }
    checkCUDNN(cudnnSetActivationDescriptor(m->actiDesc, mode,
                                            CUDNN_PROPAGATE_NAN, 0.0));
    Domain input_domain, output_domain;
    input_domain.dim = sub_input.num_dims;
    for (int i = 0; i < sub_input.num_dims; i++) {
      input_domain.rect_data[i] = 0;
      input_domain.rect_data[i+input_domain.dim] = sub_input.dims[i].size-1;
    }
    output_domain.dim = sub_output.num_dims;
    for (int i = 0; i < sub_output.num_dims; i++) {
      output_domain.rect_data[i] = 0;
      output_domain.rect_data[i+input_domain.dim] = sub_output.dims[i].size-1;
    }
    checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->inputTensor, input_domain));
    checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->outputTensor, output_domain));
  }
  sim->free_all();
  float* input_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_ptr != NULL);
  float* output_ptr = NULL;
  if (inplace) {
    output_ptr = input_ptr;
  } else {
    output_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  }
  assert(output_ptr != NULL);

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(m, input_ptr, output_ptr, sub_output.get_volume());
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float* input_grad_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    assert(input_grad_ptr != NULL);
    float* output_grad_ptr = NULL;
    if (inplace) {
      output_grad_ptr = input_grad_ptr;
    } else {
      output_grad_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    }
    assert(output_grad_ptr != NULL);
    backward = [&] {
      backward_kernel(m, input_ptr, input_grad_ptr, output_ptr, output_grad_ptr,
          sub_output.get_volume());
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure Elewise Unary] name(%s) num_elements(%zu) forward_time(%.4lf) backward_time(%.4lf)\n",
        name, sub_output.get_volume(),
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Measure Elewise Unary] name(%s) num_elements(%zu) forward_time(%.4lf)\n",
        name, sub_output.get_volume(),
        cost_metrics.forward_time);
  }
  return true;
}

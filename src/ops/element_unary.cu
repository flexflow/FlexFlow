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

#include "flexflow/ops/element_unary.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {
// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::Rect;
using Legion::PhysicalRegion;
using Legion::coord_t;

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
      case OP_SCALAR_ADD:
      {
	out[i] = in[i] + scalar;
	break;
      }
      case OP_SCALAR_SUB:
      {
	out[i] = in[i] - scalar;
	break;
      }
      case OP_SCALAR_TRUE_DIV:
      {
	out[i] = in[i] / scalar;
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
                                  size_t num_elements, 
                                  cudaStream_t stream)
{
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  if (use_cudnn(m->op_type)) {
    checkCUDNN(cudnnActivationForward(m->handle.dnn, m->actiDesc,
        &alpha, m->inputTensor, input_ptr,
        &beta, m->outputTensor, output_ptr));
  } else {
    elewise_unary_forward_kernel<<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS, 0, stream>>>(
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

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  forward_kernel(m, input_ptr, output_ptr, input_domain.get_volume(), stream);
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
      case OP_SCALAR_ADD:
      {
	input_grad[i] = output_grad[i];
	break;
      }
      case OP_SCALAR_SUB:
      {
	input_grad[i] = output_grad[i];
	break;
      }
      case OP_SCALAR_TRUE_DIV:
      {
	input_grad[i] = output_grad[i]/scalar;
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
                                   size_t num_elements,
                                   cudaStream_t stream)
{
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f;
  if (use_cudnn(m->op_type)) {
    checkCUDNN(cudnnActivationBackward(m->handle.dnn, m->actiDesc,
        &alpha, m->outputTensor, output_ptr, m->outputTensor, output_grad_ptr,
        m->inputTensor, input_ptr, &alpha, m->inputTensor, input_grad_ptr));
  } else {
    elewise_unary_backward_kernel<<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS, 0, stream>>>(
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

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  backward_kernel(m, input_ptr, input_grad_ptr, output_ptr, output_grad_ptr, input_domain.get_volume(), stream);
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
  ParallelTensorBase sub_output, sub_input;
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

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(m, input_ptr, output_ptr, sub_output.get_volume(), stream);
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
          sub_output.get_volume(), stream);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    log_measure.debug("[Measure Elewise Unary] name(%s) num_elements(%zu) forward_time(%.4lf) backward_time(%.4lf)\n",
        name, sub_output.get_volume(),
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    log_measure.debug("[Measure Elewise Unary] name(%s) num_elements(%zu) forward_time(%.4lf)\n",
        name, sub_output.get_volume(),
        cost_metrics.forward_time);
  }
  return true;
}

}; // namespace FlexFlow

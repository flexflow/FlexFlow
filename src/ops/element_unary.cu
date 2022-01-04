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
  m->data_type = eu->outputs[0]->data_type;
  // Current assume input and output have the same data type
  assert(eu->outputs[0]->data_type == eu->inputs[0]->data_type);
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

template<typename T>
__global__
void elewise_unary_forward_kernel(coord_t volume,
                                  const T scalar,
                                  OperatorType type,
                                  const T* in,
                                  T* out)
{
  CUDA_KERNEL_LOOP(i, volume)
  {
    switch (type) {
      case OP_EXP:
      {
        out[i] = (T) exp((float)in[i]);
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
	out[i] = (T)(in[i] * 0.5 * erfc(-in[i]*M_SQRT1_2));
	break;
      }
      case OP_RSQRT:
      {
        out[i] = (T)(1.0f / sqrt((float)in[i]));
	break;
      }
      case OP_POW:
      {
        out[i] = (T)(powf(in[i], scalar));
        break;
      }
      default:
        assert(false);
    }
  }
}

/*static*/
template<typename T>
void ElementUnary::forward_kernel(const ElementUnaryMeta* m,
                                  const T* input_ptr,
                                  T* output_ptr,
                                  size_t num_elements, 
                                  cudaStream_t stream)
{
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  if (use_cudnn(m->op_type)) {
    float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnActivationForward(m->handle.dnn, m->actiDesc,
        &alpha, m->inputTensor, input_ptr,
        &beta, m->outputTensor, output_ptr));
  } else {
    elewise_unary_forward_kernel<<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS, 0, stream>>>(
        num_elements, (T)m->scalar, m->op_type, input_ptr, output_ptr);
  }
}

void ElementUnary::forward_task(
    const Task* task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime* runtime)
{
  const ElementUnaryMeta* m = *((ElementUnaryMeta**) task->local_args);
  if (m->data_type == DT_FLOAT) {
    forward_task_with_type<float>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_DOUBLE) {
    forward_task_with_type<double>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_INT32) {
    forward_task_with_type<int32_t>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_INT64) {
    forward_task_with_type<int64_t>(task, regions, ctx, runtime);
  } else {
    assert(false && "Unsupported data type in Embedding forward");
  }
}

/*
  regions[0](I): input
  regions[1](O): output
*/
template<typename DT>
void ElementUnary::forward_task_with_type(
    const Task* task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime* runtime)
{
  //const ElementUnary* ele = (const ElementUnary*) task->args;
  const ElementUnaryMeta* m = *((ElementUnaryMeta**) task->local_args);
  Domain input_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  const DT* input_ptr = NULL;
  DT* output_ptr = NULL;
  if (m->inplace) {
    assert(regions.size() == 1);
    assert(task->regions.size() == 1);
    output_ptr = helperGetTensorPointerRW<DT>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
    input_ptr = output_ptr;
  } else {
    assert(regions.size() == 2);
    assert(task->regions.size() == 2);
    Domain output_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
    assert(output_domain == input_domain);
    input_ptr = helperGetTensorPointerRO<DT>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
    output_ptr = helperGetTensorPointerWO<DT>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  }

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  forward_kernel(m, input_ptr, output_ptr, input_domain.get_volume(), stream);
}

template<typename T>
__global__
void elewise_unary_backward_kernel(coord_t volume,
				   const T scalar,
                                   OperatorType type,
                                   const T* output,
                                   const T* output_grad,
                                   const T* input,
                                   T* input_grad)
{
  CUDA_KERNEL_LOOP(i, volume)
  {
    switch (type) {
      case OP_EXP:
      {
        //TODO: change to use output instead of recomputing
        input_grad[i] += (T)(output_grad[i] * exp((float)input[i]));
        break;
      }
      case OP_IDENTITY:
      {
	input_grad[i] += output_grad[i];
	break;
      } 
      case OP_SCALAR_MULTIPLY:
      {
	input_grad[i] += output_grad[i]*scalar;
	break;
      }
      case OP_SCALAR_ADD:
      {
	input_grad[i] += output_grad[i];
	break;
      }
      case OP_SCALAR_SUB:
      {
	input_grad[i] += output_grad[i];
	break;
      }
      case OP_SCALAR_TRUE_DIV:
      {
	input_grad[i] += output_grad[i]/scalar;
	break;
      }
      case OP_GELU:
      {
	input_grad[i] = (T)(output_grad[i]*(0.5 * erfc(-input[i]*M_SQRT1_2)-0.5*M_SQRT1_2*input[i]*exp(-input[i]*input[i]*0.5)));
	break;
      }
      case OP_RSQRT:
      {
        input_grad[i] = (T)(-0.5f * output_grad[i] * output[i] * output[i] * output[i]);
	break;
      }
      case OP_POW:
      {
        input_grad[i] = (T)(output_grad[i] * scalar * powf(input[i], scalar - 1));
      }
      default:
        assert(false);
    }
  }
}

/*static*/
template<typename DT>
void ElementUnary::backward_kernel(const ElementUnaryMeta* m,
                                   const DT* input_ptr,
                                   DT* input_grad_ptr,
                                   const DT* output_ptr,
                                   const DT* output_grad_ptr,
                                   size_t num_elements,
                                   cudaStream_t stream)
{
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  if (use_cudnn(m->op_type)) {
    float alpha = 1.0f;
    checkCUDNN(cudnnActivationBackward(m->handle.dnn, m->actiDesc,
        &alpha, m->outputTensor, output_ptr, m->outputTensor, output_grad_ptr,
        m->inputTensor, input_ptr, &alpha, m->inputTensor, input_grad_ptr));
  } else {
    elewise_unary_backward_kernel<DT><<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS, 0, stream>>>(
        num_elements, m->scalar, m->op_type, output_ptr, output_grad_ptr, input_ptr, input_grad_ptr);
  }
}

void ElementUnary::backward_task(
    const Task* task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime* runtime)
{
  const ElementUnaryMeta* m = *((ElementUnaryMeta**) task->local_args);
  if (m->data_type == DT_FLOAT) {
    backward_task_with_type<float>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_DOUBLE) {
    backward_task_with_type<double>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_INT32) {
    backward_task_with_type<int32_t>(task, regions, ctx, runtime);
  } else if (m->data_type == DT_INT64) {
    backward_task_with_type<int64_t>(task, regions, ctx, runtime);
  } else {
    assert(false && "Unsupported data type in Embedding forward");
  }
}

/*
  regions[0](I): input
  regions[1](I/O): input_grad
  regions[2](I): output
  regions[3](I): output_grad
*/
template<typename DT>
void ElementUnary::backward_task_with_type(
    const Task* task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime* runtime)
{
  //const ElementUnary* ele = (const ElementUnary*) task->args;
  const ElementUnaryMeta* m = *((ElementUnaryMeta**) task->local_args);
  const DT* input_ptr = NULL, *output_ptr = NULL, *output_grad_ptr = NULL;
  DT* input_grad_ptr = NULL;
  Domain input_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  if (m->inplace) {
    assert(regions.size() == 2);
    assert(task->regions.size() == 2);
    Domain input_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
    assert(input_grad_domain == input_domain);
    input_ptr = helperGetTensorPointerRO<DT>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
    input_grad_ptr = helperGetTensorPointerRW<DT>(
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
    input_ptr = helperGetTensorPointerRO<DT>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
    input_grad_ptr = helperGetTensorPointerRW<DT>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
    output_ptr = helperGetTensorPointerRO<DT>(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
    output_grad_ptr = helperGetTensorPointerRO<DT>(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);
  }

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  backward_kernel<DT>(m, input_ptr, input_grad_ptr, output_ptr, output_grad_ptr, input_domain.get_volume(), stream);
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

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

#include "flexflow/ops/element_binary.h"
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

bool ElementBinary::can_inplace_output(void)
{
  if (op_type == OP_EW_ADD)
    return false;
  if (op_type == OP_EW_MUL)
    return false;
  return false;
}

bool ElementBinary::has_inplace_output(void)
{
  return inplace_a;
}

void ElementBinary::do_inplace_output(void)
{
  inplace_a = true;
}

__host__
OpMeta* ElementBinary::init_task(const Task* task,
                                 const std::vector<PhysicalRegion> &regions,
                                 Context ctx, Runtime* runtime)
{
  ElementBinary* eb = (ElementBinary*) task->args;
  FFHandler handle = *((FFHandler*) task->local_args);
  ElementBinaryMeta* m = new ElementBinaryMeta(handle);
  m->op_type = eb->op_type;
  m->profiling = eb->profiling;
  m->inplace_a = eb->inplace_a;
  Domain input_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain output_domain;
  if (m->inplace_a) {
    assert(regions.size() == 2);
    assert(task->regions.size() == regions.size());
    output_domain = runtime->get_index_space_domain(
        ctx, task->regions[1].region.get_index_space());
    assert(output_domain == input_domain);
  } else {
    assert(regions.size() == 3);
    assert(task->regions.size() == regions.size());
    output_domain = runtime->get_index_space_domain(
        ctx, task->regions[2].region.get_index_space());
    assert(output_domain == input_domain);
  }
  cudnnOpTensorOp_t mode;
  switch (eb->op_type) {
    case OP_EW_ADD:
    case OP_EW_SUB:
      mode = CUDNN_OP_TENSOR_ADD;
      break;
    case OP_EW_MUL:
      mode = CUDNN_OP_TENSOR_MUL;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnSetOpTensorDescriptor(m->opDesc, mode,
      CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->inputTensor, input_domain));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->outputTensor, output_domain));
  return m;
}

__global__
void elewise_binary_forward_kernel(coord_t volume,
                                   const float alpha,
                                   const float beta,
                                   OperatorType type,
                                   const float* in1,
                                   const float* in2,
                                   float* out)
{
  switch (type) {
    case OP_EW_ADD:
    {
      CUDA_KERNEL_LOOP(i, volume)
      {
        out[i] = alpha * (in1[i] + in2[i]) + beta * out[i];
      }
      break;
    }
    case OP_EW_SUB:
    {
      CUDA_KERNEL_LOOP(i, volume)
      {
        out[i] = alpha * (in1[i] - in2[i]) + beta * out[i];
      }
      break;
    }
    case OP_EW_MUL:
    {
      CUDA_KERNEL_LOOP(i, volume)
      {
        out[i] = alpha * in1[i] * in2[i] + beta * out[i];
      }
      break;
    }
    case OP_EW_DIV:
    {
      CUDA_KERNEL_LOOP(i, volume)
      {
        out[i] = alpha * (in1[i] / in2[i]) + beta * out[i];
      }
      break;
    }
    default:
      assert(false);
  }
}

/*static*/
void ElementBinary::forward_kernel(const ElementBinaryMeta* m,
                                   const float* in1_ptr,
                                   const float* in2_ptr,
                                   float* out_ptr,
                                   cudaStream_t stream)
{
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;
  switch (m->op_type) {
    case OP_EW_SUB:
      alpha2 = -1.0f;
      break;
    case OP_EW_ADD:
    case OP_EW_MUL:
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnOpTensor(m->handle.dnn, m->opDesc,
      &alpha1, m->inputTensor, in1_ptr,
      &alpha2, m->inputTensor, in2_ptr,
      &beta, m->outputTensor, out_ptr));
}

/*
  regions[0](I): in1
  regions[1](I): in2
  regions[2](O): output
*/
__host__
void ElementBinary::forward_task(const Task* task,
                                 const std::vector<PhysicalRegion> &regions,
                                 Context ctx, Runtime* runtime)
{
  //const ElementBinary* ele = (const ElementBinary*) task->args;
  const ElementBinaryMeta* m = *((ElementBinaryMeta**) task->local_args);
  Domain in1_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain in2_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  assert(in1_domain == in2_domain);
  const float* in1_ptr = NULL, *in2_ptr = NULL;
  float *out_ptr = NULL;
  if (m->inplace_a) {
    assert(regions.size() == 2);
    assert(task->regions.size() == 2);
    out_ptr = helperGetTensorPointerRW<float>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    in2_ptr = helperGetTensorPointerRO<float>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    in1_ptr = out_ptr;
  } else {
    assert(regions.size() == 3);
    assert(task->regions.size() == 3);
    Domain out_domain = runtime->get_index_space_domain(
        ctx, task->regions[2].region.get_index_space());
    assert(out_domain == in1_domain);
    in1_ptr = helperGetTensorPointerRO<float>(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    in2_ptr = helperGetTensorPointerRO<float>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    out_ptr = helperGetTensorPointerWO<float>(
        regions[2], task->regions[2], FID_DATA, ctx, runtime);
  }

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  //print_tensor<float>(in1_ptr, in1_domain.get_volume(), "input1:");
  //print_tensor<float>(in2_ptr, in2_domain.get_volume(), "input2:");
  forward_kernel(m, in1_ptr, in2_ptr, out_ptr, stream);
  //print_tensor<float>(out_ptr, in1_domain.get_volume(), "output:");
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    char const *opName;
    switch (m->op_type) {
      case OP_EW_ADD:
        opName = "Add";
        break;
      case OP_EW_SUB:
        opName = "Sub";
        break;
      case OP_EW_MUL:
        opName = "Mul";
        break;
      case OP_EW_DIV:
        opName = "Div";
        break;
      default:
        assert(false);
    }
    log_measure.debug("[%s] forward time (CF) = %.2fms\n", opName, elapsed);
  }
}

__global__
void elewise_binary_backward_kernel(coord_t volume,
                                    const float alpha,
                                    const float beta,
                                    OperatorType type,
                                    const float* out_grad,
                                    const float* in1,
                                    const float* in2,
                                    float* in1_grad,
                                    float* in2_grad)
{
  CUDA_KERNEL_LOOP(i, volume)
  {
    switch (type) {
      case OP_EW_ADD:
      {
        in1_grad[i] = alpha * out_grad[i] + beta * in1_grad[i];
        in2_grad[i] = alpha * out_grad[i] + beta * in2_grad[i];
        break;
      }
      case OP_EW_SUB:
      {
        in1_grad[i] = alpha * out_grad[i] + beta * in1_grad[i];
        in2_grad[i] = - alpha * out_grad[i] + beta * in2_grad[i];
        break;
      }
      case OP_EW_MUL:
      {
        in1_grad[i] = alpha * out_grad[i] * in2[i] + beta * in1_grad[i];
        in2_grad[i] = alpha * out_grad[i] * in1[i] + beta * in2_grad[i];
        break;
      }
      case OP_EW_DIV:
      {
        in1_grad[i] = alpha * out_grad[i] / in2[i] + beta * in1_grad[i];
        in2_grad[i] = - alpha * out_grad[i] * in1[i] / (in2[i] * in2[i]) + beta * in2_grad[i];
        break;
      }
      default:
        assert(false);
    }
  }
}

/*static*/
void ElementBinary::backward_kernel(const ElementBinaryMeta* m,
                                    const float* out_grad_ptr,
                                    const float* in1_ptr,
                                    const float* in2_ptr,
                                    float* in1_grad_ptr,
                                    float* in2_grad_ptr,
                                    cudaStream_t stream)
{
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha1 = 1.0f, alpha2 = 1.0f, beta = 1.0f;
  switch (m->op_type) {
    case OP_EW_ADD:
      alpha1 = 1.0f;
      alpha2 = 0.0f;
      break;
    case OP_EW_SUB:
      alpha1 = -1.0f;
      alpha2 = 0.0f;
      break;
    case OP_EW_MUL:
      alpha1 = 1.0f;
      alpha2 = 1.0f;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnOpTensor(m->handle.dnn, m->opDesc,
      &alpha1, m->outputTensor, out_grad_ptr,
      &alpha2, m->inputTensor, in1_ptr,
      &beta, m->inputTensor, in2_grad_ptr));
  switch (m->op_type) {
    case OP_EW_ADD:
    case OP_EW_SUB:
      alpha1 = 1.0f;
      alpha2 = 0.0f;
      break;
    case OP_EW_MUL:
      alpha1 = 1.0f;
      alpha2 = 1.0f;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnOpTensor(m->handle.dnn, m->opDesc,
      &alpha1, m->outputTensor, out_grad_ptr,
      &alpha2, m->inputTensor, in2_ptr,
      &beta, m->inputTensor, in1_grad_ptr));
}

/*
  regions[0](I or I/O): out_grad (I/O if inplace_a)
  regions[1](I): in0
  regions[2](I/O): in0_grad (Missing if in0_grad = out_grad)
  regions[3](I): in1 (Missing if in0 = in1)
  regions[4](I/O): in1_grad (Missing if in0=in1)
*/
void ElementBinary::backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime* runtime)
{
  //const ElementBinary* ele = (const ElementBinary*) task->args;
  const ElementBinaryMeta* m = *((ElementBinaryMeta**) task->local_args);
  const float *in0_ptr = NULL, *in1_ptr = NULL, *out_grad_ptr = NULL;
  float *in0_grad_ptr = NULL, *in1_grad_ptr = NULL;
  Domain out_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  if (m->inplace_a) {
    in0_grad_ptr = helperGetTensorPointerRW<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
    if (regions.size() == 2 || regions.size() == 4);
    assert(task->regions.size() == regions.size());
    if (regions.size() == 2) {
      Domain in0_domain = runtime->get_index_space_domain(
        ctx, task->regions[1].region.get_index_space());
      assert(in0_domain == out_grad_domain);
      in0_ptr = helperGetTensorPointerRO<float>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
      in1_ptr = in0_ptr;
      in1_grad_ptr = in0_grad_ptr;
      out_grad_ptr = in0_grad_ptr;
    } else {
      Domain in0_domain = runtime->get_index_space_domain(
        ctx, task->regions[1].region.get_index_space());
      Domain in1_domain = runtime->get_index_space_domain(
        ctx, task->regions[2].region.get_index_space());
      assert(in0_domain == out_grad_domain);
      assert(in1_domain == out_grad_domain);
      in0_ptr = helperGetTensorPointerRO<float>(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
      in1_ptr = helperGetTensorPointerRO<float>(
        regions[2], task->regions[2], FID_DATA, ctx, runtime);
      in1_grad_ptr = helperGetTensorPointerRW<float>(
        regions[3], task->regions[3], FID_DATA, ctx, runtime);
      out_grad_ptr = in0_grad_ptr;
    }
  } else {
    assert(regions.size() == 3 || regions.size() == 5);
    assert(task->regions.size() == regions.size());
    out_grad_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
    Domain in0_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
    Domain in0_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
    assert(out_grad_domain == in0_grad_domain);
    assert(out_grad_domain == in0_domain);
    in0_ptr = helperGetTensorPointerRO<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
    in0_grad_ptr = helperGetTensorPointerRW<float>(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
    if (regions.size() == 3) {
      // in0 == in1
      in1_ptr = in0_ptr;
      in1_grad_ptr = in0_grad_ptr;
    } else {
      Domain in1_domain = runtime->get_index_space_domain(
        ctx, task->regions[3].region.get_index_space());
      Domain in1_grad_domain = runtime->get_index_space_domain(
        ctx, task->regions[4].region.get_index_space());
      assert(out_grad_domain == in1_domain);
      assert(out_grad_domain == in1_grad_domain);
      in1_ptr = helperGetTensorPointerRO<float>(
        regions[3], task->regions[3], FID_DATA, ctx, runtime);
      in1_grad_ptr = helperGetTensorPointerRW<float>(
        regions[4], task->regions[4], FID_DATA, ctx, runtime);
    }
  }

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  backward_kernel(m, out_grad_ptr, in0_ptr, in1_ptr, in0_grad_ptr, in1_grad_ptr, stream);
  //elewise_binary_backward_kernel<<<GET_BLOCKS(out_grad_domain.get_volume()), CUDA_NUM_THREADS>>>(
    //out_grad_domain.get_volume(), alpha, alpha, ele->op_type, out_grad_ptr, in1_ptr, in2_ptr,
    //in1_grad_ptr, in2_grad_ptr);
}

ElementBinaryMeta::ElementBinaryMeta(FFHandler handler)
: OpMeta(handler)
{
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateOpTensorDescriptor(&opDesc));
  op_type = OP_NOOP;
}

bool ElementBinary::measure_operator_cost(Simulator* sim,
                                          const ParallelConfig& pc,
                                          CostMetrics& cost_metrics) const
{
  ParallelTensorBase sub_output, sub_input1, sub_input0;
  if (!outputs[0]->get_output_sub_tensor(pc, sub_output, op_type))
    return false;
  if (!inputs[0]->get_input_sub_tensor(pc, sub_input0, op_type))
    return false;
  if (!inputs[1]->get_input_sub_tensor(pc, sub_input1, op_type))
    return false;
  ElementBinaryMeta* m = sim->ele_binary_meta;
  m->op_type = op_type;
  cudnnOpTensorOp_t mode;
  switch (op_type) {
    case OP_EW_ADD:
    case OP_EW_SUB:
      mode = CUDNN_OP_TENSOR_ADD;
      break;
    case OP_EW_MUL:
      mode = CUDNN_OP_TENSOR_MUL;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnSetOpTensorDescriptor(m->opDesc, mode,
      CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));
  Domain input_domain = sub_input0.get_domain();
  Domain output_domain = sub_output.get_domain();
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->inputTensor, input_domain));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->outputTensor, output_domain));
  sim->free_all();
  float* input0_ptr = (float*)sim->allocate(sub_input0.get_volume(), DT_FLOAT);
  assert(input0_ptr != NULL);
  float* input1_ptr = (float*)sim->allocate(sub_input1.get_volume(), DT_FLOAT);
  assert(input1_ptr != NULL);
  float* output_ptr = NULL;
  if (inplace_a) {
    output_ptr = input0_ptr;
  } else {
    output_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  }
  assert(output_ptr != NULL);

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(m, input0_ptr, input1_ptr, output_ptr, stream);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float* input0_grad_ptr = (float*)sim->allocate(sub_input0.get_volume(), DT_FLOAT);
    assert(input0_grad_ptr != NULL);
    float* input1_grad_ptr = (float*)sim->allocate(sub_input0.get_volume(), DT_FLOAT);
    assert(input1_grad_ptr != NULL);
    float* output_grad_ptr = NULL;
    if (inplace_a) {
      output_grad_ptr = input0_grad_ptr;
    } else {
      output_grad_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    }
    assert(output_grad_ptr != NULL);
    backward = [&] {
      backward_kernel(m, output_grad_ptr, input0_ptr, input1_ptr, input0_grad_ptr, input1_grad_ptr, stream);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    log_measure.debug("[Measure Elewise Binary] name(%s) num_elements(%zu) forward_time(%.4lf) backward_time(%.4lf)\n",
        name, sub_output.get_volume(),
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    log_measure.debug("[Measure Elewise Binary] name(%s) num_elements(%zu) forward_time(%.4lf)\n",
        name, sub_output.get_volume(),
        cost_metrics.forward_time);
  }

  return true;
}

}; // namespace FlexFlow

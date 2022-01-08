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

#include <hip/hip_runtime.h>
#include "flexflow/ops/element_binary.h"
#include "flexflow/utils/hip_helper.h"

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
  if (op_type == OP_EW_ADD || op_type == OP_EW_MUL) {
    // TODO: Currently assume that we always inplace_a
    if (outputs[0]->num_dims != inputs[0]->num_dims)
      return false;
    for (int i = 0; i < inputs[0]->num_dims; i++) {
      if (inputs[0]->dims[i] != outputs[0]->dims[i])
        return false;
    }
    return true;
  }
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
  m->has_same_operands = eb->has_same_operands;
  Domain input1_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain input2_domain, output_domain;
  size_t num_regions = 1;
  if (!m->has_same_operands) {
    input2_domain = runtime->get_index_space_domain(
        ctx, task->regions[num_regions].region.get_index_space());
    num_regions ++;
  } else {
    input2_domain = input1_domain;
  }
  if (!m->inplace_a) {
    output_domain = runtime->get_index_space_domain(
        ctx, task->regions[num_regions].region.get_index_space());
    num_regions ++;
    // check that input can broadcast to output
    for (int i = 0; i < output_domain.dim; i++) {
      int output_dim_size = output_domain.hi()[i] - output_domain.lo()[i] + 1;
      if (i < input1_domain.dim) {
        int input1_dim_size = input1_domain.hi()[i] - input1_domain.lo()[i] + 1;
        assert(input1_dim_size == output_dim_size || input1_dim_size == 1);
      }
      if (i < input2_domain.dim) {
        int input2_dim_size = input2_domain.hi()[i] - input2_domain.lo()[i] + 1;
        assert(input2_dim_size == output_dim_size || input2_dim_size == 1);
      }
    }
  } else {
    output_domain = input1_domain;
  }
  assert(task->regions.size() == regions.size());
  assert(regions.size() == num_regions);
#if 0
  hipdnnOpTensorOp_t mode;
  switch (eb->op_type) {
    case OP_EW_ADD:
    case OP_EW_SUB:
      mode = HIPDNN_OP_TENSOR_ADD;
      break;
    case OP_EW_MUL:
      mode = HIPDNN_OP_TENSOR_MUL;
      break;
    default:
      assert(false);
  }
  checkCUDNN(hipdnnSetOpTensorDescriptor(m->opDesc, mode,
      HIPDNN_DATA_FLOAT, HIPDNN_PROPAGATE_NAN));
  checkCUDNN(hipdnnSetReduceTensorDescriptor(m->reduceAddDesc, HIPDNN_REDUCE_TENSOR_ADD,
      HIPDNN_DATA_FLOAT, HIPDNN_PROPAGATE_NAN, HIPDNN_REDUCE_TENSOR_NO_INDICES, HIPDNN_32BIT_INDICES));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->input1Tensor, input1_domain));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->input2Tensor, input2_domain));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->outputTensor, output_domain));
#endif  
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
                                   hipStream_t stream)
{
#if 0
  checkCUDA(hipblasSetStream(m->handle.blas, stream));
  checkCUDNN(hipdnnSetStream(m->handle.dnn, stream));

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
  checkCUDNN(hipdnnOpTensor(m->handle.dnn, m->opDesc,
      &alpha1, m->input1Tensor, in1_ptr,
      &alpha2, m->input2Tensor, in2_ptr,
      &beta, m->outputTensor, out_ptr));
#endif
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
  if (!m->has_same_operands) {
    Domain in2_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
    // Currently only support broadcast for add and sub
    if (in1_domain != in2_domain) {
      assert(m->op_type == OP_EW_SUB || m->op_type == OP_EW_ADD);
    }
  }
  const float* in1_ptr = NULL, *in2_ptr = NULL;
  float *out_ptr = NULL;
  if (m->inplace_a) {
    if (m->has_same_operands) {
      assert(regions.size() == 1);
      assert(task->regions.size() == 1);
      out_ptr = helperGetTensorPointerRW<float>(
          regions[0], task->regions[0], FID_DATA, ctx, runtime);
      in2_ptr = out_ptr;
      in1_ptr = out_ptr;
    } else {
      assert(regions.size() == 2);
      assert(task->regions.size() == 2);
      out_ptr = helperGetTensorPointerRW<float>(
          regions[0], task->regions[0], FID_DATA, ctx, runtime);
      in2_ptr = helperGetTensorPointerRO<float>(
          regions[1], task->regions[1], FID_DATA, ctx, runtime);
      in1_ptr = out_ptr;
    }
  } else {
    if (m->has_same_operands) {
      assert(regions.size() == 2);
      assert(task->regions.size() == 2);
      Domain out_domain = runtime->get_index_space_domain(
          ctx, task->regions[1].region.get_index_space());
      assert(out_domain == in1_domain);
      in1_ptr = helperGetTensorPointerRO<float>(
          regions[0], task->regions[0], FID_DATA, ctx, runtime);
      in2_ptr = in1_ptr;
      out_ptr = helperGetTensorPointerWO<float>(
          regions[1], task->regions[1], FID_DATA, ctx, runtime);
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
  }

  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    hipEventCreate(&t_start);
    hipEventCreate(&t_end);
    hipEventRecord(t_start, stream);
  }
  //print_tensor<float>(in1_ptr, in1_domain.get_volume(), "input1:");
  //print_tensor<float>(in2_ptr, in2_domain.get_volume(), "input2:");
  forward_kernel(m, in1_ptr, in2_ptr, out_ptr, stream);
  //print_tensor<float>(out_ptr, in1_domain.get_volume(), "output:");
  if (m->profiling) {
    hipEventRecord(t_end, stream);
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    hipEventDestroy(t_start);
    hipEventDestroy(t_end);
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
                                    hipStream_t stream)
{
#if 0
  checkCUDA(hipblasSetStream(m->handle.blas, stream));
  checkCUDNN(hipdnnSetStream(m->handle.dnn, stream));

  int output_ndims, input_ndims;
  int output_dims[MAX_TENSOR_DIM], input_dims[MAX_TENSOR_DIM];
  int output_strides[MAX_TENSOR_DIM], input_strides[MAX_TENSOR_DIM];
  hipdnnDataType_t output_datatype, input_datatype;
  checkCUDNN(hipdnnGetTensorNdDescriptor(m->outputTensor, 4,
      &output_datatype, &output_ndims, output_dims, output_strides));

  if (m->op_type == OP_EW_ADD || m->op_type == OP_EW_SUB) {
    float alpha = 1.0f, beta = 1.0f;
    checkCUDNN(hipdnnGetTensorNdDescriptor(m->input1Tensor, 4,
        &input_datatype, &input_ndims, input_dims, input_strides));
    bool has_reduce = false;
    assert(input_ndims == output_ndims);
    for (int i = 0; i < input_ndims; i++)
      if (input_dims[i] != output_dims[i])
        has_reduce = true;
    if (has_reduce) {
      checkCUDNN(hipdnnReduceTensor(m->handle.dnn, m->reduceAddDesc,
          nullptr/*indices*/, 0/*indicesSizeInBytes*/,
          m->handle.workSpace, m->handle.workSpaceSize,
          &alpha, m->outputTensor, out_grad_ptr,
          &beta, m->input1Tensor, in1_grad_ptr));
    } else {
      checkCUDNN(hipdnnAddTensor(m->handle.dnn,
          &alpha, m->outputTensor, out_grad_ptr,
          &beta, m->input1Tensor, in1_grad_ptr));
    }
    if (m->op_type == OP_EW_SUB)
      alpha = -1.0f;
    checkCUDNN(hipdnnGetTensorNdDescriptor(m->input2Tensor, 4,
        &input_datatype, &input_ndims, input_dims, input_strides));
    has_reduce = false;
    assert(input_ndims == output_ndims);
    for (int i = 0; i < input_ndims; i++)
      if (input_dims[i] != output_dims[i])
        has_reduce = true;
    if (has_reduce) {
      checkCUDNN(hipdnnReduceTensor(m->handle.dnn, m->reduceAddDesc,
          nullptr/*indices*/, 0/*indicesSizeInBytes*/,
          m->handle.workSpace, m->handle.workSpaceSize,
          &alpha, m->outputTensor, out_grad_ptr,
          &beta, m->input2Tensor, in2_grad_ptr));
    } else {
      checkCUDNN(hipdnnAddTensor(m->handle.dnn,
          &alpha, m->outputTensor, out_grad_ptr,
          &beta, m->input2Tensor, in2_grad_ptr));
    }
  } else if (m->op_type == OP_EW_MUL) {
    float alpha1 = 1.0f, alpha2 = 1.0f, beta = 1.0f;
    checkCUDNN(hipdnnOpTensor(m->handle.dnn, m->opDesc,
        &alpha1, m->outputTensor, out_grad_ptr,
        &alpha2, m->input2Tensor, in2_ptr,
        &beta, m->input1Tensor, in1_grad_ptr));
    checkCUDNN(hipdnnOpTensor(m->handle.dnn, m->opDesc,
        &alpha1, m->outputTensor, out_grad_ptr,
        &alpha2, m->input2Tensor, in1_ptr,
        &beta, m->input1Tensor, in2_grad_ptr));
  } else {
    assert(false && "Unsupported ElementWise Binary Type");
  }
#endif
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
    assert(regions.size() == 2 || regions.size() == 4);
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
      //assert(in1_domain == out_grad_domain);
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
      //assert(out_grad_domain == in1_domain);
      assert(in1_domain == in1_grad_domain);
      in1_ptr = helperGetTensorPointerRO<float>(
        regions[3], task->regions[3], FID_DATA, ctx, runtime);
      in1_grad_ptr = helperGetTensorPointerRW<float>(
        regions[4], task->regions[4], FID_DATA, ctx, runtime);
    }
  }

  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  hipEvent_t t_start, t_end;
  if (m->profiling) {
    hipEventCreate(&t_start);
    hipEventCreate(&t_end);
    hipEventRecord(t_start, stream);
  }

  backward_kernel(m, out_grad_ptr, in0_ptr, in1_ptr, in0_grad_ptr, in1_grad_ptr, stream);
  //elewise_binary_backward_kernel<<<GET_BLOCKS(out_grad_domain.get_volume()), CUDA_NUM_THREADS>>>(
    //out_grad_domain.get_volume(), alpha, alpha, ele->op_type, out_grad_ptr, in1_ptr, in2_ptr,
    //in1_grad_ptr, in2_grad_ptr);
  if (m->profiling) {
    hipEventRecord(t_end, stream);
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    hipEventDestroy(t_start);
    hipEventDestroy(t_end);
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
    printf("[%s] backward time (CB) = %.2fms\n", opName, elapsed);
  }

}

ElementBinaryMeta::ElementBinaryMeta(FFHandler handler)
: OpMeta(handler)
{
#if 0
  checkCUDNN(hipdnnCreateTensorDescriptor(&input1Tensor));
  checkCUDNN(hipdnnCreateTensorDescriptor(&input2Tensor));
  checkCUDNN(hipdnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(hipdnnCreateOpTensorDescriptor(&opDesc));
  checkCUDNN(hipdnnCreateReduceTensorDescriptor(&reduceAddDesc));
  op_type = OP_NOOP;
#endif
}

bool ElementBinary::measure_operator_cost(Simulator* sim,
                                          const ParallelConfig& pc,
                                          CostMetrics& cost_metrics) const
{
#if 0
  ParallelTensorBase sub_output, sub_input1, sub_input2;
  if (!outputs[0]->get_output_sub_tensor(pc, sub_output, op_type))
    return false;
  if (!inputs[0]->get_input_sub_tensor(pc, sub_input1, op_type))
    return false;
  if (!inputs[1]->get_input_sub_tensor(pc, sub_input2, op_type))
    return false;
  ElementBinaryMeta* m = sim->ele_binary_meta;
  m->op_type = op_type;
  hipdnnOpTensorOp_t mode;
  switch (op_type) {
    case OP_EW_ADD:
    case OP_EW_SUB:
      mode = HIPDNN_OP_TENSOR_ADD;
      break;
    case OP_EW_MUL:
      mode = HIPDNN_OP_TENSOR_MUL;
      break;
    default:
      assert(false);
  }
  checkCUDNN(hipdnnSetOpTensorDescriptor(m->opDesc, mode,
      HIPDNN_DATA_FLOAT, HIPDNN_PROPAGATE_NAN));
  Domain input1_domain = sub_input1.get_domain();
  Domain input2_domain = sub_input2.get_domain();
  Domain output_domain = sub_output.get_domain();
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->input1Tensor, input1_domain));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->input2Tensor, input2_domain));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(m->outputTensor, output_domain));
  sim->free_all();
  float* input1_ptr = (float*)sim->allocate(sub_input1.get_volume(), DT_FLOAT);
  assert(input1_ptr != NULL);
  float* input2_ptr = (float*)sim->allocate(sub_input2.get_volume(), DT_FLOAT);
  assert(input2_ptr != NULL);
  float* output_ptr = NULL;
  if (inplace_a) {
    output_ptr = input1_ptr;
  } else {
    output_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  }
  assert(output_ptr != NULL);

  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(m, input1_ptr, input2_ptr, output_ptr, stream);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float* input1_grad_ptr = (float*)sim->allocate(sub_input1.get_volume(), DT_FLOAT);
    assert(input1_grad_ptr != NULL);
    float* input2_grad_ptr = (float*)sim->allocate(sub_input2.get_volume(), DT_FLOAT);
    assert(input2_grad_ptr != NULL);
    float* output_grad_ptr = NULL;
    if (inplace_a) {
      output_grad_ptr = input1_grad_ptr;
    } else {
      output_grad_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    }
    assert(output_grad_ptr != NULL);
    backward = [&] {
      backward_kernel(m, output_grad_ptr, input1_ptr, input2_ptr, input1_grad_ptr, input2_grad_ptr, stream);
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
#endif

  return true;
}

}; // namespace FlexFlow

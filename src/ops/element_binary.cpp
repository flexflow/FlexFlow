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
using Legion::Domain;
using Legion::coord_t;

/*static*/
void ElementBinary::init_kernel(ElementBinaryMeta* m,
                                const Domain& input1_domain,
                                const Domain& input2_domain,
                                const Domain& output_domain)
{
#if 0
  hipdnnOpTensorOp_t mode;
  switch (m->op_type) {
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

/*static*/
void ElementBinary::forward_kernel_wrapper(const ElementBinaryMeta* m,
                                           const float* in1_ptr,
                                           const float* in2_ptr,
                                           float* out_ptr)
{
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
  ElementBinary::forward_kernel(m, in1_ptr, in2_ptr, out_ptr);
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

/*static*/
void ElementBinary::backward_kernel_wrapper(const ElementBinaryMeta* m,
                                            const float* out_grad_ptr,
                                            const float* in1_ptr,
                                            const float* in2_ptr,
                                            float* in1_grad_ptr,
                                            float* in2_grad_ptr)
{
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  hipEvent_t t_start, t_end;
  if (m->profiling) {
    hipEventCreate(&t_start);
    hipEventCreate(&t_end);
    hipEventRecord(t_start, stream);
  }

  ElementBinary::backward_kernel(m, out_grad_ptr, in1_ptr, in2_ptr, in1_grad_ptr, in2_grad_ptr, stream);
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

}; // namespace FlexFlow

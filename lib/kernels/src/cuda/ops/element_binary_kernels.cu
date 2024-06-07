/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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

#include "device.h"
#include "kernels/element_binary_kernels.h"
#include "kernels/ff_handle.h"
#include "op-attrs/datatype.h"
#include "op-attrs/operator_type.h"

namespace FlexFlow {
namespace Kernels {
namespace ElementBinary {

__global__ void elewise_binary_backward_kernel(size_t volume,
                                               float const alpha,
                                               float const beta,
                                               OperatorType type,
                                               float const *out_grad,
                                               float const *lhs,
                                               float const *rhs,
                                               float *lhs_grad,
                                               float *rhs_grad) {
  CUDA_KERNEL_LOOP(i, volume) {
    switch (type) {
      case OperatorType::EW_ADD: {
        lhs_grad[i] = alpha * out_grad[i] + beta * lhs_grad[i];
        rhs_grad[i] = alpha * out_grad[i] + beta * rhs_grad[i];
        break;
      }
      case OperatorType::EW_SUB: {
        lhs_grad[i] = alpha * out_grad[i] + beta * lhs_grad[i];
        rhs_grad[i] = -alpha * out_grad[i] + beta * rhs_grad[i];
        break;
      }
      case OperatorType::EW_MUL: {
        lhs_grad[i] = alpha * out_grad[i] * rhs[i] + beta * lhs_grad[i];
        rhs_grad[i] = alpha * out_grad[i] * lhs[i] + beta * rhs_grad[i];
        break;
      }
      case OperatorType::EW_DIV: {
        lhs_grad[i] = alpha * out_grad[i] / rhs[i] + beta * lhs_grad[i];
        rhs_grad[i] = -alpha * out_grad[i] * lhs[i] / (rhs[i] * rhs[i]) +
                      beta * rhs_grad[i];
        break;
      }
      case OperatorType::EW_MAX: {
        lhs_grad[i] = (lhs[i] >= rhs[i])
                          ? alpha * out_grad[i] + beta * lhs_grad[i]
                          : beta * lhs_grad[i];
        rhs_grad[i] = (rhs[i] >= lhs[i])
                          ? alpha * out_grad[i] + beta * rhs_grad[i]
                          : beta * rhs_grad[i];
        break;
      }
      case OperatorType::EW_MIN: {
        lhs_grad[i] = (lhs[i] <= rhs[i])
                          ? alpha * out_grad[i] + beta * lhs_grad[i]
                          : beta * lhs_grad[i];
        rhs_grad[i] = (rhs[i] <= lhs[i])
                          ? alpha * out_grad[i] + beta * rhs_grad[i]
                          : beta * rhs_grad[i];
        break;
      }
      default:
        assert(false);
    }
  }
}

ElementBinaryPerDeviceState init_kernel(PerDeviceFFHandle handle,
                                        OperatorType op_type,
                                        bool should_broadcast_lhs,
                                        bool should_broadcast_rhs,
                                        ArrayShape lhs_shape,
                                        ArrayShape rhs_shape,
                                        ArrayShape output_shape) {
  ffTensorDescriptor_t inputLHSTensor;
  ffTensorDescriptor_t inputRHSTensor;
  ffTensorDescriptor_t outputTensor;
  ffOpTensorDescriptor_t opDesc;
  ffReduceTensorDescriptor_t reduceAddDesc;
  cudnnOpTensorOp_t mode;

  checkCUDNN(cudnnCreateTensorDescriptor(&inputLHSTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&inputRHSTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateOpTensorDescriptor(&opDesc));
  checkCUDNN(cudnnCreateReduceTensorDescriptor(&reduceAddDesc));

  switch (op_type) {
    case OperatorType::EW_ADD:
    case OperatorType::EW_SUB:
      mode = CUDNN_OP_TENSOR_ADD;
      break;
    case OperatorType::EW_MUL:
      mode = CUDNN_OP_TENSOR_MUL;
      break;
    case OperatorType::EW_MAX:
      mode = CUDNN_OP_TENSOR_MAX;
      break;
    case OperatorType::EW_MIN:
      mode = CUDNN_OP_TENSOR_MIN;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnSetOpTensorDescriptor(
      opDesc, mode, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));
  checkCUDNN(cudnnSetReduceTensorDescriptor(reduceAddDesc,
                                            CUDNN_REDUCE_TENSOR_ADD,
                                            CUDNN_DATA_FLOAT,
                                            CUDNN_PROPAGATE_NAN,
                                            CUDNN_REDUCE_TENSOR_NO_INDICES,
                                            CUDNN_32BIT_INDICES));
  checkCUDNN(cudnnSetTensorDescriptorFromArrayShape(inputLHSTensor, lhs_shape));
  checkCUDNN(cudnnSetTensorDescriptorFromArrayShape(inputRHSTensor, rhs_shape));
  checkCUDNN(
      cudnnSetTensorDescriptorFromArrayShape(outputTensor, output_shape));

  ElementBinaryPerDeviceState per_device_state = {handle,
                                                  inputLHSTensor,
                                                  inputRHSTensor,
                                                  outputTensor,
                                                  opDesc,
                                                  reduceAddDesc};
  return per_device_state;
}

void forward_kernel(cudaStream_t stream,
                    ElementBinaryPerDeviceState const &m,
                    float const *lhs_ptr,
                    float const *rhs_ptr,
                    float *out_ptr,
                    OperatorType op_type,
                    bool broadcast_inputLHS,
                    PerDeviceFFHandle handle) {
  checkCUBLAS(cublasSetStream(handle.blas, stream));
  checkCUDNN(cudnnSetStream(handle.dnn, stream));
  float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;
  switch (op_type) {
    case OperatorType::EW_SUB:
      alpha2 = -1.0f;
      break;
    case OperatorType::EW_ADD:
    case OperatorType::EW_MUL:
    case OperatorType::EW_MAX:
    case OperatorType::EW_MIN:
      break;
    default:
      assert(false);
  }
  // cudnn currently does not support broadcasting the first input in
  // cudnnOpTensor
  if (broadcast_inputLHS) {
    // currently only handle add and sub
    assert(op_type == OperatorType::EW_SUB || op_type == OperatorType::EW_ADD ||
           op_type == OperatorType::EW_MUL);
    if (op_type == OperatorType::EW_SUB || op_type == OperatorType::EW_ADD) {
      // output = (beta*output + alpha1*input1) + beta*output = input1
      checkCUDNN(cudnnOpTensor(handle.dnn,
                               m.opDesc,
                               &beta,
                               m.outputTensor,
                               out_ptr,
                               &alpha1,
                               m.inputLHSTensor,
                               lhs_ptr,
                               &beta,
                               m.outputTensor,
                               out_ptr));
      // output = (beta*output + alpha2*input2) + alpha1*output = alpha2*input2
      // + alpha1*input1
      checkCUDNN(cudnnOpTensor(handle.dnn,
                               m.opDesc,
                               &beta,
                               m.outputTensor,
                               out_ptr,
                               &alpha2,
                               m.inputRHSTensor,
                               rhs_ptr,
                               &alpha1,
                               m.outputTensor,
                               out_ptr));
    } else if (op_type == OperatorType::EW_MUL) {
      checkCUDNN(cudnnSetOpTensorDescriptor(m.opDesc,
                                            CUDNN_OP_TENSOR_ADD,
                                            CUDNN_DATA_FLOAT,
                                            CUDNN_PROPAGATE_NAN));
      // output = (beta*output + alpha1*input1) + beta*output = input1
      checkCUDNN(cudnnOpTensor(handle.dnn,
                               m.opDesc,
                               &beta,
                               m.outputTensor,
                               out_ptr,
                               &alpha1,
                               m.inputLHSTensor,
                               lhs_ptr,
                               &beta,
                               m.outputTensor,
                               out_ptr));
      checkCUDNN(cudnnSetOpTensorDescriptor(m.opDesc,
                                            CUDNN_OP_TENSOR_MUL,
                                            CUDNN_DATA_FLOAT,
                                            CUDNN_PROPAGATE_NAN));
      // output = (alpha1*output * alpha2*input2) + beta*output
      checkCUDNN(cudnnOpTensor(handle.dnn,
                               m.opDesc,
                               &alpha1,
                               m.outputTensor,
                               out_ptr,
                               &alpha2,
                               m.inputRHSTensor,
                               rhs_ptr,
                               &beta,
                               m.outputTensor,
                               out_ptr));
    }
  } else {
    checkCUDNN(cudnnOpTensor(handle.dnn,
                             m.opDesc,
                             &alpha1,
                             m.inputLHSTensor,
                             lhs_ptr,
                             &alpha2,
                             m.inputRHSTensor,
                             rhs_ptr,
                             &beta,
                             m.outputTensor,
                             out_ptr));
  }
}

void backward_kernel(cudaStream_t stream,
                     ElementBinaryPerDeviceState const &m,
                     float const *out_grad_ptr,
                     float const *lhs_ptr,
                     float const *rhs_ptr,
                     float *lhs_grad_ptr,
                     float *rhs_grad_ptr,
                     OperatorType op_type,
                     bool broadcast_inputLHS,
                     bool broadcast_inputRHS,
                     PerDeviceFFHandle handle) {
  checkCUBLAS(cublasSetStream(handle.blas, stream));
  checkCUDNN(cudnnSetStream(handle.dnn, stream));

  if (op_type == OperatorType::EW_ADD || op_type == OperatorType::EW_SUB) {
    float alpha = 1.0f, beta = 1.0f;
    if (lhs_grad_ptr != nullptr) {
      if (broadcast_inputLHS) {
        checkCUDNN(cudnnReduceTensor(handle.dnn,
                                     m.reduceAddDesc,
                                     nullptr /*indices*/,
                                     0 /*indicesSizeInBytes*/,
                                     handle.workSpace,
                                     handle.workSpaceSize,
                                     &alpha,
                                     m.outputTensor,
                                     out_grad_ptr,
                                     &beta,
                                     m.inputLHSTensor,
                                     lhs_grad_ptr));
      } else {
        checkCUDNN(cudnnAddTensor(handle.dnn,
                                  &alpha,
                                  m.outputTensor,
                                  out_grad_ptr,
                                  &beta,
                                  m.inputLHSTensor,
                                  lhs_grad_ptr));
      }
    }
    if (op_type == OperatorType::EW_SUB) {
      alpha = -1.0f;
    }
    if (rhs_grad_ptr != nullptr) {
      if (broadcast_inputRHS) {
        checkCUDNN(cudnnReduceTensor(handle.dnn,
                                     m.reduceAddDesc,
                                     nullptr /*indices*/,
                                     0 /*indicesSizeInBytes*/,
                                     handle.workSpace,
                                     handle.workSpaceSize,
                                     &alpha,
                                     m.outputTensor,
                                     out_grad_ptr,
                                     &beta,
                                     m.inputRHSTensor,
                                     rhs_grad_ptr));
      } else {
        checkCUDNN(cudnnAddTensor(handle.dnn,
                                  &alpha,
                                  m.outputTensor,
                                  out_grad_ptr,
                                  &beta,
                                  m.inputRHSTensor,
                                  rhs_grad_ptr));
      }
    }
  } else if (op_type == OperatorType::EW_MUL) {
    float alpha1 = 1.0f, alpha2 = 1.0f, beta = 1.0f, zero = 0.0f;
    if (lhs_grad_ptr != nullptr) {
      if (broadcast_inputLHS) {
        checkCUDNN(cudnnOpTensor(handle.dnn,
                                 m.opDesc,
                                 &alpha1,
                                 m.outputTensor,
                                 out_grad_ptr,
                                 &alpha2,
                                 m.inputRHSTensor,
                                 rhs_ptr,
                                 &zero,
                                 m.outputTensor,
                                 handle.workSpace));
        checkCUDNN(cudnnReduceTensor(
            handle.dnn,
            m.reduceAddDesc,
            nullptr /*indices*/,
            0 /*indicesSizeInBytes*/,
            (void *)((char *)handle.workSpace + sizeof(*out_grad_ptr)),
            handle.workSpaceSize - sizeof(*out_grad_ptr),
            &alpha1,
            m.outputTensor,
            handle.workSpace,
            &beta,
            m.inputLHSTensor,
            lhs_grad_ptr));
      } else {
        checkCUDNN(cudnnOpTensor(handle.dnn,
                                 m.opDesc,
                                 &alpha1,
                                 m.outputTensor,
                                 out_grad_ptr,
                                 &alpha2,
                                 m.inputRHSTensor,
                                 rhs_ptr,
                                 &beta,
                                 m.inputLHSTensor,
                                 lhs_grad_ptr));
      }
    }
    if (rhs_grad_ptr != nullptr) {
      if (broadcast_inputRHS) {
        checkCUDNN(cudnnOpTensor(handle.dnn,
                                 m.opDesc,
                                 &alpha1,
                                 m.outputTensor,
                                 out_grad_ptr,
                                 &alpha2,
                                 m.inputLHSTensor,
                                 lhs_ptr,
                                 &zero,
                                 m.outputTensor,
                                 handle.workSpace));
        checkCUDNN(cudnnReduceTensor(
            handle.dnn,
            m.reduceAddDesc,
            nullptr /*indices*/,
            0 /*indicesSizeInBytes*/,
            (void *)((char *)handle.workSpace + sizeof(*out_grad_ptr)),
            handle.workSpaceSize - sizeof(*out_grad_ptr),
            &alpha1,
            m.outputTensor,
            handle.workSpace,
            &beta,
            m.inputRHSTensor,
            rhs_grad_ptr));
      } else {
        checkCUDNN(cudnnOpTensor(handle.dnn,
                                 m.opDesc,
                                 &alpha1,
                                 m.outputTensor,
                                 out_grad_ptr,
                                 &alpha2,
                                 m.inputLHSTensor,
                                 lhs_ptr,
                                 &beta,
                                 m.inputRHSTensor,
                                 rhs_grad_ptr));
      }
    }
  } else if (op_type == OperatorType::EW_MIN ||
             op_type == OperatorType::EW_MAX) {
    float alpha = 1.0f, beta = 1.0f;
    cudnnDataType_t dataType;
    int n;
    int dims[MAX_TENSOR_DIM];
    int strides[MAX_TENSOR_DIM];
    checkCUDNN(cudnnGetTensorNdDescriptor(
        m.outputTensor, MAX_TENSOR_DIM, &dataType, &n, dims, strides));
    size_t volume = 1;
    for (int i = 0; i < n; i++) {
      volume *= dims[i];
    }
    elewise_binary_backward_kernel<<<GET_BLOCKS(volume),
                                     CUDA_NUM_THREADS,
                                     0,
                                     stream>>>(volume,
                                               alpha,
                                               beta,
                                               op_type,
                                               out_grad_ptr,
                                               lhs_ptr,
                                               rhs_ptr,
                                               lhs_grad_ptr,
                                               rhs_grad_ptr);
  } else {
    assert(false && "Unsupported ElementWise Binary Type");
  }
}

} // namespace ElementBinary
} // namespace Kernels
} // namespace FlexFlow

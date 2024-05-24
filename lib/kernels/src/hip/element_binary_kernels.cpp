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

#include "kernels/element_binary_kernels.h"
#include "device.h"
#include "kernels/ff_handle.h"
#include "op-attrs/datatype.h"
#include "op-attrs/op.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {
namespace Kernels {
namespace ElementBinary {

using OperatorType = Op;

__global__ void elewise_binary_backward_kernel(coord_t volume,
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
      case Op::EW_ADD: {
        lhs_grad[i] = alpha * out_grad[i] + beta * lhs_grad[i];
        rhs_grad[i] = alpha * out_grad[i] + beta * rhs_grad[i];
        break;
      }
      case Op::EW_SUB: {
        lhs_grad[i] = alpha * out_grad[i] + beta * lhs_grad[i];
        rhs_grad[i] = -alpha * out_grad[i] + beta * rhs_grad[i];
        break;
      }
      case Op::EW_MUL: {
        lhs_grad[i] = alpha * out_grad[i] * rhs[i] + beta * lhs_grad[i];
        rhs_grad[i] = alpha * out_grad[i] * lhs[i] + beta * rhs_grad[i];
        break;
      }
      case Op::EW_DIV: {
        lhs_grad[i] = alpha * out_grad[i] / rhs[i] + beta * lhs_grad[i];
        rhs_grad[i] = -alpha * out_grad[i] * lhs[i] / (rhs[i] * rhs[i]) +
                      beta * rhs_grad[i];
        break;
      }
      case Op::EW_MAX: {
        lhs_grad[i] = (lhs[i] >= rhs[i])
                          ? alpha * out_grad[i] + beta * lhs_grad[i]
                          : beta * lhs_grad[i];
        rhs_grad[i] = (rhs[i] >= lhs[i])
                          ? alpha * out_grad[i] + beta * rhs_grad[i]
                          : beta * rhs_grad[i];
        break;
      }
      case Op::EW_MIN: {
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
  miopenTensorOp_t mode;

  checkCUDNN(miopenCreateTensorDescriptor(&inputLHSTensor));
  checkCUDNN(miopenCreateTensorDescriptor(&inputRHSTensor));
  checkCUDNN(miopenCreateTensorDescriptor(&outputTensor));
  checkCUDNN(miopenCreateOpTensorDescriptor(&opDesc));
  checkCUDNN(miopenCreateReduceTensorDescriptor(&reduceAddDesc));

  switch (op_type) {
    case Op::EW_ADD:
    case Op::EW_SUB:
      mode = miopenTensorOpAdd;
      break;
    case Op::EW_MUL:
      mode = miopenTensorOpMul;
      break;
    case Op::EW_MAX:
      mode = miopenOpTensorMax;
      break;
    case Op::EW_MIN:
      mode = miopenOpTensorMin;
      break;
    default:
      assert(false);
  }
  checkCUDNN(miopenSetOpTensorDescriptor(
      opDesc, mode, miopenFloat, MIOPEN_PROPAGATE_NAN));
  checkCUDNN(miopenSetReduceTensorDescriptor(reduceAddDesc,
                                             MIOPEN_REDUCE_TENSOR_ADD,
                                             miopenFloat,
                                             MIOPEN_PROPAGATE_NAN,
                                             MIOPEN_REDUCE_TENSOR_NO_INDICES,
                                             MIOPEN_32BIT_INDICES));
  checkCUDNN(
      miopenSetTensorDescriptorFromArrayShape(inputLHSTensor, lhs_shape));
  checkCUDNN(
      miopenSetTensorDescriptorFromArrayShape(inputRHSTensor, rhs_shape));
  checkCUDNN(
      miopenSetTensorDescriptorFromArrayShape(outputTensor, output_shape));

  ElementBinaryPerDeviceState per_device_state = {handle,
                                                  inputLHSTensor,
                                                  inputRHSTensor,
                                                  outputTensor,
                                                  opDesc,
                                                  reduceAddDesc};
  return per_device_state;
}

void forward_kernel(hipStream_t stream,
                    ElementBinaryPerDeviceState const &m,
                    float const *lhs_ptr,
                    float const *rhs_ptr,
                    float *out_ptr,
                    OperatorType op_type,
                    bool broadcast_inputLHS,
                    PerDeviceFFHandle handle) {
  checkCUDA(hipblasSetStream(handle.blas, stream));
  checkCUDNN(miopenSetStream(handle.dnn, stream));

  float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;
  switch (op_type) {
    case Op::EW_SUB:
      alpha2 = -1.0f;
      break;
    case Op::EW_ADD:
    case Op::EW_MUL:
    case Op::EW_MAX:
    case Op::EW_MIN:
      break;
    default:
      assert(false);
  }
  // cudnn currently does not support broadcasting the first input in
  // cudnnOpTensor
  if (broadcast_inputLHS) {
    // currently only handle add and sub
    assert(op_type == Op::EW_SUB || op_type == Op::EW_ADD ||
           op_type == Op::EW_MUL);
    if (op_type == Op::EW_SUB || op_type == Op::EW_ADD) {
      checkCUDNN(miopenOpTensor(handle.dnn,
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
      checkCUDNN(miopenOpTensor(handle.dnn,
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
    } else if (op_type == Op::EW_MUL) {
      checkCUDNN(cudnnSetOpTensorDescriptor(m.opDesc,
                                            CUDNN_OP_TENSOR_MUL,
                                            CUDNN_DATA_FLOAT,
                                            CUDNN_NOT_PROPAGATE_NAN));
      checkCUDNN(miopenOpTensor(handle.dnn,
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
      checkCUDNN(cudnnSetOpTensorDescriptor(m.opDesc,
                                            CUDNN_OP_TENSOR_ADD,
                                            CUDNN_DATA_FLOAT,
                                            CUDNN_NOT_PROPAGATE_NAN));

      checkCUDNN(miopenOpTensor(handle.dnn,
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
    } else {
      checkCUDNN(miopenOpTensor(handle.dnn,
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
}

void backward_kernel(hipStream_t stream,
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
  checkCUDA(hipblasSetStream(handle.blas, stream));
  checkCUDNN(miopenSetStream(handle.dnn, stream));

  if (m.op_type == Op::EW_ADD || m.op_type == Op::EW_SUB) {
    float alpha = 1.0f, beta = 1.0f;
    if (lhs_grad_ptr != nullptr) {
      if (broadcast_inputLHS) {
        checkCUDNN(miopenReduceTensor(handle.dnn,
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
        checkCUDNN(miopenOpTensor(handle.dnn,
                                  miopenTensorOpAdd,
                                  &alpha,
                                  m.outputTensor,
                                  out_grad_ptr,
                                  &alpha,
                                  m.outputTensor,
                                  out_grad_ptr,
                                  &beta,
                                  m.inputLHSTensor,
                                  lhs_grad_ptr));
      }
    }
    if (m.op_type == Op::EW_SUB) {
      alpha = -1.0f;
    }
    if (rhs_grad_ptr != nullptr) {
      if (broadcast_inputRHS) {
        checkCUDNN(miopenReduceTensor(handle.dnn,
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
        checkCUDNN(miopenOpTensor(handle.dnn,
                                  miopenTensorOpAdd,
                                  &alpha,
                                  m.outputTensor,
                                  out_grad_ptr,
                                  &alpha,
                                  m.outputTensor,
                                  out_grad_ptr,
                                  &beta,
                                  m.inputRHSTensor,
                                  rhs_grad_ptr));
      }
    }
  } else if (m.op_type == Op::EW_MUL) {
    float alpha1 = 1.0f, alpha2 = 1.0f, beta = 1.0f, zero = 0.0f;
    if (lhs_grad_ptr != nullptr) {
      if (broadcast_inputLHS) {
        checkCUDNN(miopenOpTensor(handle.dnn,
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
        checkCUDNN(miopenReduceTensor(handle.dnn,
                                      m.reduceAddDesc,
                                      nullptr /*indices*/,
                                      0 /*indicesSizeInBytes*/,
                                      handle.workSpace,
                                      handle.workSpaceSize,
                                      &alpha1,
                                      m.outputTensor,
                                      out_grad_ptr,
                                      &zero,
                                      m.inputLHSTensor,
                                      lhs_grad_ptr));
      } else {
        checkCUDNN(miopenOpTensor(handle.dnn,
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
        checkCUDNN(miopenOpTensor(handle.dnn,
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
        checkCUDNN(miopenReduceTensor(handle.dnn,
                                      m.reduceAddDesc,
                                      nullptr /*indices*/,
                                      0 /*indicesSizeInBytes*/,
                                      handle.workSpace,
                                      handle.workSpaceSize,
                                      &alpha1,
                                      m.outputTensor,
                                      out_grad_ptr,
                                      &zero,
                                      m.inputRHSTensor,
                                      rhs_grad_ptr));
      } else {
        checkCUDNN(miopenOpTensor(handle.dnn,
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
  } else if (op_type == Op::EW_MIN || op_type == Op::EW_MAX) {
    float alpha = 1.0f, beta = 1.0f;
    miopenDataType_t data_type;
    int n;
    int dims[MAX_TENSOR_DIM];
    int strides[MAX_TENSOR_DIM];
    checkCUDNN(miopenGetTensorDescriptorSize(m.outputTensor, &n));
    size_t volume = 1;
    for (int i = 0; i < n; i++) {
      volume *= dims[i];
    }
    // launch hip kernel
    hipLaunchKernelGGL(elewise_binary_backward_kernel,
                       GET_BLOCKS(volume),
                       CUDA_NUM_THREADS,
                       0,
                       stream,
                       volume,
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

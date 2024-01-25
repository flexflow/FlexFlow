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
#include "kernels/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {
namespace Kernels {
namespace ElementBinary {

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
      mode = miopenTensorOpMax;
      break;
    case Op::EW_MIN:
      mode = miopenTensorOpMin;
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

__global__ void elewise_binary_forward_kernel(coord_t volume,
                                              float const alpha,
                                              float const beta,
                                              OperatorType type,
                                              float const *in1,
                                              float const *in2,
                                              float *out) {
  switch (type) {
    case Op::EW_ADD: {
      CUDA_KERNEL_LOOP(i, volume) {
        out[i] = alpha * (in1[i] + in2[i]) + beta * out[i];
      }
      break;
    }
    case Op::EW_SUB: {
      CUDA_KERNEL_LOOP(i, volume) {
        out[i] = alpha * (in1[i] - in2[i]) + beta * out[i];
      }
      break;
    }
    case Op::EW_MUL: {
      CUDA_KERNEL_LOOP(i, volume) {
        out[i] = alpha * in1[i] * in2[i] + beta * out[i];
      }
      break;
    }
    case Op::EW_DIV: {
      CUDA_KERNEL_LOOP(i, volume) {
        out[i] = alpha * (in1[i] / in2[i]) + beta * out[i];
      }
      break;
    }
    default:
      assert(false);
  }
}

__global__ void elewise_binary_backward_kernel(coord_t volume,
                                               float const alpha,
                                               float const beta,
                                               OperatorType type,
                                               float const *out_grad,
                                               float const *in1,
                                               float const *in2,
                                               float *in1_grad,
                                               float *in2_grad) {
  CUDA_KERNEL_LOOP(i, volume) {
    switch (type) {
      case Op::EW_ADD: {
        in1_grad[i] = alpha * out_grad[i] + beta * in1_grad[i];
        in2_grad[i] = alpha * out_grad[i] + beta * in2_grad[i];
        break;
      }
      case Op::EW_SUB: {
        in1_grad[i] = alpha * out_grad[i] + beta * in1_grad[i];
        in2_grad[i] = -alpha * out_grad[i] + beta * in2_grad[i];
        break;
      }
      case Op::EW_MUL: {
        in1_grad[i] = alpha * out_grad[i] * in2[i] + beta * in1_grad[i];
        in2_grad[i] = alpha * out_grad[i] * in1[i] + beta * in2_grad[i];
        break;
      }
      case Op::EW_DIV: {
        in1_grad[i] = alpha * out_grad[i] / in2[i] + beta * in1_grad[i];
        in2_grad[i] = -alpha * out_grad[i] * in1[i] / (in2[i] * in2[i]) +
                      beta * in2_grad[i];
        break;
      }
      default:
        assert(false);
    }
  }
}

void forward_kernel(hipStream_t stream,
                    ElementBinaryPerDeviceState const *m,
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
      break;
    default:
      assert(false);
  }
  // cudnn currently does not support broadcasting the first input in
  // cudnnOpTensor
  if (broadcast_inputLHS) {
    // currently only handle add and sub
    assert(op_type == Op::EW_SUB || op_type == Op::EW_ADD);
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
    float alpha = 1.0f, alpha2 = 0.0f, beta = 1.0f;
    if (lhs_grad_ptr != nullptr) {
      if (m.broadcast_input1) {
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
                                  &alpha2,
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
      if (m.broadcast_input2) {
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
                                  &alpha2,
                                  m.outputTensor,
                                  out_grad_ptr,
                                  &beta,
                                  m.inputRHSTensor,
                                  rhs_grad_ptr));
      }
    }
  } else if (m.op_type == Op::EW_MUL) {
    float alpha1 = 1.0f, alpha2 = 1.0f, beta = 1.0f;
    if (lhs_grad_ptr != nullptr) {
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
    if (rhs_grad_ptr != nullptr) {
      checkCUDNN(miopenOpTensor(handle.dnn,
                                m.opDesc,
                                &alpha1,
                                m.outputTensor,
                                out_grad_ptr,
                                &alpha2,
                                m.inputRHSTensor,
                                lhs_ptr,
                                &beta,
                                m.inputLHSTensor,
                                rhs_grad_ptr));
    }
  } else {
    assert(false && "Unsupported ElementWise Binary Type");
  }
}

} // namespace ElementBinary
} // namespace Kernels
} // namespace FlexFlow

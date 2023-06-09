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
// declare Legion names
using Legion::coord_t;
using Legion::Domain;

ElementBinaryPerDeviceState::ElementBinaryPerDeviceState(FFHandler handler)
    : OpPerDeviceState(handler) {
  checkCUDNN(miopenCreateTensorDescriptor(&input1Tensor));
  checkCUDNN(miopenCreateTensorDescriptor(&input2Tensor));
  checkCUDNN(miopenCreateTensorDescriptor(&outputTensor));
  checkCUDNN(miopenCreateReduceTensorDescriptor(&reduceAddDesc));
  op_type = OP_NOOP;
}

namespace Kernels {
namespace ElementBinary {

/*static*/
void init_kernel(ElementBinaryPerDeviceState *m,
                 Domain const &input1_domain,
                 Domain const &input2_domain,
                 Domain const &output_domain) {
  miopenTensorOp_t mode;
  switch (m->op_type) {
    case OP_EW_ADD:
    case OP_EW_SUB:
      mode = miopenTensorOpAdd;
      break;
    case OP_EW_MUL:
      mode = miopenTensorOpMul;
      break;
    default:
      assert(false);
  }
  m->opDesc = mode;
  checkCUDNN(miopenSetReduceTensorDescriptor(m->reduceAddDesc,
                                             MIOPEN_REDUCE_TENSOR_ADD,
                                             miopenFloat,
                                             MIOPEN_PROPAGATE_NAN,
                                             MIOPEN_REDUCE_TENSOR_NO_INDICES,
                                             MIOPEN_32BIT_INDICES));
  checkCUDNN(
      cudnnSetTensorDescriptorFromDomain(m->input1Tensor, input1_domain));
  checkCUDNN(
      cudnnSetTensorDescriptorFromDomain(m->input2Tensor, input2_domain));
  checkCUDNN(
      cudnnSetTensorDescriptorFromDomain(m->outputTensor, output_domain));
}

__global__ void elewise_binary_forward_kernel(coord_t volume,
                                              float const alpha,
                                              float const beta,
                                              OperatorType type,
                                              float const *in1,
                                              float const *in2,
                                              float *out) {
  switch (type) {
    case OP_EW_ADD: {
      CUDA_KERNEL_LOOP(i, volume) {
        out[i] = alpha * (in1[i] + in2[i]) + beta * out[i];
      }
      break;
    }
    case OP_EW_SUB: {
      CUDA_KERNEL_LOOP(i, volume) {
        out[i] = alpha * (in1[i] - in2[i]) + beta * out[i];
      }
      break;
    }
    case OP_EW_MUL: {
      CUDA_KERNEL_LOOP(i, volume) {
        out[i] = alpha * in1[i] * in2[i] + beta * out[i];
      }
      break;
    }
    case OP_EW_DIV: {
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
      case OP_EW_ADD: {
        in1_grad[i] = alpha * out_grad[i] + beta * in1_grad[i];
        in2_grad[i] = alpha * out_grad[i] + beta * in2_grad[i];
        break;
      }
      case OP_EW_SUB: {
        in1_grad[i] = alpha * out_grad[i] + beta * in1_grad[i];
        in2_grad[i] = -alpha * out_grad[i] + beta * in2_grad[i];
        break;
      }
      case OP_EW_MUL: {
        in1_grad[i] = alpha * out_grad[i] * in2[i] + beta * in1_grad[i];
        in2_grad[i] = alpha * out_grad[i] * in1[i] + beta * in2_grad[i];
        break;
      }
      case OP_EW_DIV: {
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
                    float const *in1_ptr,
                    float const *in2_ptr,
                    float *out_ptr) {
  checkCUDA(hipblasSetStream(m->handle.blas, stream));
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

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
  // cudnn currently does not support broadcasting the first input in
  // cudnnOpTensor
  if (m->broadcast_input1) {
    // currently only handle add and sub
    assert(m->op_type == OP_EW_SUB || m->op_type == OP_EW_ADD);
    checkCUDNN(miopenOpTensor(m->handle.dnn,
                              m->opDesc,
                              &beta,
                              m->outputTensor,
                              out_ptr,
                              &alpha1,
                              m->input1Tensor,
                              in1_ptr,
                              &beta,
                              m->outputTensor,
                              out_ptr));
    checkCUDNN(miopenOpTensor(m->handle.dnn,
                              m->opDesc,
                              &beta,
                              m->outputTensor,
                              out_ptr,
                              &alpha2,
                              m->input2Tensor,
                              in2_ptr,
                              &alpha1,
                              m->outputTensor,
                              out_ptr));
  } else {
    checkCUDNN(miopenOpTensor(m->handle.dnn,
                              m->opDesc,
                              &alpha1,
                              m->input1Tensor,
                              in1_ptr,
                              &alpha2,
                              m->input2Tensor,
                              in2_ptr,
                              &beta,
                              m->outputTensor,
                              out_ptr));
  }
}

void backward_kernel(hipStream_t stream,
                     ElementBinaryPerDeviceState const *m,
                     float const *out_grad_ptr,
                     float const *in1_ptr,
                     float const *in2_ptr,
                     float *in1_grad_ptr,
                     float *in2_grad_ptr) {
  checkCUDA(hipblasSetStream(m->handle.blas, stream));
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

  if (m->op_type == OP_EW_ADD || m->op_type == OP_EW_SUB) {
    float alpha = 1.0f, alpha2 = 0.0f, beta = 1.0f;
    if (in1_grad_ptr != nullptr) {
      if (m->broadcast_input1) {
        checkCUDNN(miopenReduceTensor(m->handle.dnn,
                                      m->reduceAddDesc,
                                      nullptr /*indices*/,
                                      0 /*indicesSizeInBytes*/,
                                      m->handle.workSpace,
                                      m->handle.workSpaceSize,
                                      &alpha,
                                      m->outputTensor,
                                      out_grad_ptr,
                                      &beta,
                                      m->input1Tensor,
                                      in1_grad_ptr));
      } else {
        checkCUDNN(miopenOpTensor(m->handle.dnn,
                                  miopenTensorOpAdd,
                                  &alpha,
                                  m->outputTensor,
                                  out_grad_ptr,
                                  &alpha2,
                                  m->outputTensor,
                                  out_grad_ptr,
                                  &beta,
                                  m->input1Tensor,
                                  in1_grad_ptr));
      }
    }
    if (m->op_type == OP_EW_SUB) {
      alpha = -1.0f;
    }
    if (in2_grad_ptr != nullptr) {
      if (m->broadcast_input2) {
        checkCUDNN(miopenReduceTensor(m->handle.dnn,
                                      m->reduceAddDesc,
                                      nullptr /*indices*/,
                                      0 /*indicesSizeInBytes*/,
                                      m->handle.workSpace,
                                      m->handle.workSpaceSize,
                                      &alpha,
                                      m->outputTensor,
                                      out_grad_ptr,
                                      &beta,
                                      m->input2Tensor,
                                      in2_grad_ptr));
      } else {
        checkCUDNN(miopenOpTensor(m->handle.dnn,
                                  miopenTensorOpAdd,
                                  &alpha,
                                  m->outputTensor,
                                  out_grad_ptr,
                                  &alpha2,
                                  m->outputTensor,
                                  out_grad_ptr,
                                  &beta,
                                  m->input2Tensor,
                                  in2_grad_ptr));
      }
    }
  } else if (m->op_type == OP_EW_MUL) {
    float alpha1 = 1.0f, alpha2 = 1.0f, beta = 1.0f;
    if (in1_grad_ptr != nullptr) {
      checkCUDNN(miopenOpTensor(m->handle.dnn,
                                m->opDesc,
                                &alpha1,
                                m->outputTensor,
                                out_grad_ptr,
                                &alpha2,
                                m->input2Tensor,
                                in2_ptr,
                                &beta,
                                m->input1Tensor,
                                in1_grad_ptr));
    }
    if (in2_grad_ptr != nullptr) {
      checkCUDNN(miopenOpTensor(m->handle.dnn,
                                m->opDesc,
                                &alpha1,
                                m->outputTensor,
                                out_grad_ptr,
                                &alpha2,
                                m->input2Tensor,
                                in1_ptr,
                                &beta,
                                m->input1Tensor,
                                in2_grad_ptr));
    }
  } else {
    assert(false && "Unsupported ElementWise Binary Type");
  }
}

} // namespace ElementBinary
} // namespace Kernels
} // namespace FlexFlow

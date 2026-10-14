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

#include "internal/device.h"
#include "kernels/element_binary_kernels_gpu.h"
#include "kernels/ff_handle.h"
#include "op-attrs/datatype.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

__global__ void elewise_binary_backward_kernel(size_t volume,
                                               float const alpha,
                                               float const beta,
                                               ElementBinaryOp type,
                                               float const *out_grad,
                                               float const *lhs,
                                               float const *rhs,
                                               float *lhs_grad,
                                               float *rhs_grad) {
  CUDA_KERNEL_LOOP(i, volume) {
    switch (type) {
      case ElementBinaryOp::ADD: {
        lhs_grad[i] = alpha * out_grad[i] + beta * lhs_grad[i];
        rhs_grad[i] = alpha * out_grad[i] + beta * rhs_grad[i];
        break;
      }
      case ElementBinaryOp::SUBTRACT: {
        lhs_grad[i] = alpha * out_grad[i] + beta * lhs_grad[i];
        rhs_grad[i] = -alpha * out_grad[i] + beta * rhs_grad[i];
        break;
      }
      case ElementBinaryOp::MULTIPLY: {
        lhs_grad[i] = alpha * out_grad[i] * rhs[i] + beta * lhs_grad[i];
        rhs_grad[i] = alpha * out_grad[i] * lhs[i] + beta * rhs_grad[i];
        break;
      }
      case ElementBinaryOp::DIVIDE: {
        lhs_grad[i] = alpha * out_grad[i] / rhs[i] + beta * lhs_grad[i];
        rhs_grad[i] = -alpha * out_grad[i] * lhs[i] / (rhs[i] * rhs[i]) +
                      beta * rhs_grad[i];
        break;
      }
      case ElementBinaryOp::MAX: {
        lhs_grad[i] = (lhs[i] >= rhs[i])
                          ? alpha * out_grad[i] + beta * lhs_grad[i]
                          : beta * lhs_grad[i];
        rhs_grad[i] = (rhs[i] >= lhs[i])
                          ? alpha * out_grad[i] + beta * rhs_grad[i]
                          : beta * rhs_grad[i];
        break;
      }
      case ElementBinaryOp::MIN: {
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

ElementBinaryPerDeviceState
    element_binary_gpu_init_kernel(ElementBinaryAttrs const &attrs,
                                   TensorShape const &lhs_shape,
                                   TensorShape const &rhs_shape,
                                   TensorShape const &output_shape) {
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

  switch (attrs.op) {
    case ElementBinaryOp::ADD:
    case ElementBinaryOp::SUBTRACT:
      mode = CUDNN_OP_TENSOR_ADD;
      break;
    case ElementBinaryOp::MULTIPLY:
      mode = CUDNN_OP_TENSOR_MUL;
      break;
    case ElementBinaryOp::MAX:
      mode = CUDNN_OP_TENSOR_MAX;
      break;
    case ElementBinaryOp::MIN:
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
  checkCUDNN(
      cudnnSetTensorDescriptorFromTensorShape(inputLHSTensor, lhs_shape));
  checkCUDNN(
      cudnnSetTensorDescriptorFromTensorShape(inputRHSTensor, rhs_shape));
  checkCUDNN(
      cudnnSetTensorDescriptorFromTensorShape(outputTensor, output_shape));

  ElementBinaryPerDeviceState per_device_state = ElementBinaryPerDeviceState{
      /*inputLHSTensor=*/inputLHSTensor,
      /*inputRHSTensor=*/inputRHSTensor,
      /*outputTensor=*/outputTensor,
      /*opDesc=*/opDesc,
      /*reduceAddDesc=*/reduceAddDesc,
  };
  return per_device_state;
}

void element_binary_gpu_forward_kernel(
    ffStream_t stream,
    PerDeviceFFHandle const &handle,
    ElementBinaryPerDeviceState const &per_device_state,
    ElementBinaryAttrs const &attrs,
    GenericTensorAccessorR const &lhs,
    GenericTensorAccessorR const &rhs,
    GenericTensorAccessorW const &output) {
  checkCUBLAS(cublasSetStream(handle.blas, stream));
  checkCUDNN(cudnnSetStream(handle.dnn, stream));
  float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;
  switch (attrs.op) {
    case ElementBinaryOp::SUBTRACT:
      alpha2 = -1.0f;
      break;
    case ElementBinaryOp::ADD:
    case ElementBinaryOp::MULTIPLY:
    case ElementBinaryOp::MAX:
    case ElementBinaryOp::MIN:
      break;
    default:
      assert(false);
  }
  // cudnn currently does not support broadcasting the first input in
  // cudnnOpTensor
  if (attrs.should_broadcast_lhs) {
    // currently only handle add and sub
    assert(attrs.op == ElementBinaryOp::SUBTRACT ||
           attrs.op == ElementBinaryOp::ADD ||
           attrs.op == ElementBinaryOp::MULTIPLY);
    if (attrs.op == ElementBinaryOp::SUBTRACT ||
        attrs.op == ElementBinaryOp::ADD) {
      // output = (beta*output + alpha1*input1) + beta*output = input1
      checkCUDNN(cudnnOpTensor(handle.dnn,
                               per_device_state.opDesc,
                               &beta,
                               per_device_state.outputTensor,
                               output.get_float_ptr(),
                               &alpha1,
                               per_device_state.inputLHSTensor,
                               lhs.get_float_ptr(),
                               &beta,
                               per_device_state.outputTensor,
                               output.get_float_ptr()));
      // output = (beta*output + alpha2*input2) + alpha1*output = alpha2*input2
      // + alpha1*input1
      checkCUDNN(cudnnOpTensor(handle.dnn,
                               per_device_state.opDesc,
                               &beta,
                               per_device_state.outputTensor,
                               output.get_float_ptr(),
                               &alpha2,
                               per_device_state.inputRHSTensor,
                               rhs.get_float_ptr(),
                               &alpha1,
                               per_device_state.outputTensor,
                               output.get_float_ptr()));
    } else if (attrs.op == ElementBinaryOp::MULTIPLY) {
      checkCUDNN(cudnnSetOpTensorDescriptor(per_device_state.opDesc,
                                            CUDNN_OP_TENSOR_ADD,
                                            CUDNN_DATA_FLOAT,
                                            CUDNN_PROPAGATE_NAN));
      // output = (beta*output + alpha1*input1) + beta*output = input1
      checkCUDNN(cudnnOpTensor(handle.dnn,
                               per_device_state.opDesc,
                               &beta,
                               per_device_state.outputTensor,
                               output.get_float_ptr(),
                               &alpha1,
                               per_device_state.inputLHSTensor,
                               lhs.get_float_ptr(),
                               &beta,
                               per_device_state.outputTensor,
                               output.get_float_ptr()));
      checkCUDNN(cudnnSetOpTensorDescriptor(per_device_state.opDesc,
                                            CUDNN_OP_TENSOR_MUL,
                                            CUDNN_DATA_FLOAT,
                                            CUDNN_PROPAGATE_NAN));
      // output = (alpha1*output * alpha2*input2) + beta*output
      checkCUDNN(cudnnOpTensor(handle.dnn,
                               per_device_state.opDesc,
                               &alpha1,
                               per_device_state.outputTensor,
                               output.get_float_ptr(),
                               &alpha2,
                               per_device_state.inputRHSTensor,
                               rhs.get_float_ptr(),
                               &beta,
                               per_device_state.outputTensor,
                               output.get_float_ptr()));
    }
  } else {
    checkCUDNN(cudnnOpTensor(handle.dnn,
                             per_device_state.opDesc,
                             &alpha1,
                             per_device_state.inputLHSTensor,
                             lhs.get_float_ptr(),
                             &alpha2,
                             per_device_state.inputRHSTensor,
                             rhs.get_float_ptr(),
                             &beta,
                             per_device_state.outputTensor,
                             output.get_float_ptr()));
  }
}

void element_binary_gpu_backward_kernel(
    ffStream_t stream,
    PerDeviceFFHandle const &handle,
    ElementBinaryPerDeviceState const &per_device_state,
    ElementBinaryAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &lhs,
    GenericTensorAccessorW const &lhs_grad,
    GenericTensorAccessorR const &rhs,
    GenericTensorAccessorW const &rhs_grad) {
  checkCUBLAS(cublasSetStream(handle.blas, stream));
  checkCUDNN(cudnnSetStream(handle.dnn, stream));

  if (attrs.op == ElementBinaryOp::ADD ||
      attrs.op == ElementBinaryOp::SUBTRACT) {
    float alpha = 1.0f, beta = 1.0f;
    if (lhs_grad.get_float_ptr() != nullptr) {
      if (attrs.should_broadcast_lhs) {
        checkCUDNN(cudnnReduceTensor(handle.dnn,
                                     per_device_state.reduceAddDesc,
                                     nullptr /*indices*/,
                                     0 /*indicesSizeInBytes*/,
                                     handle.workSpace,
                                     handle.workSpaceSize,
                                     &alpha,
                                     per_device_state.outputTensor,
                                     output_grad.get_float_ptr(),
                                     &beta,
                                     per_device_state.inputLHSTensor,
                                     lhs_grad.get_float_ptr()));
      } else {
        checkCUDNN(cudnnAddTensor(handle.dnn,
                                  &alpha,
                                  per_device_state.outputTensor,
                                  output_grad.get_float_ptr(),
                                  &beta,
                                  per_device_state.inputLHSTensor,
                                  lhs_grad.get_float_ptr()));
      }
    }
    if (attrs.op == ElementBinaryOp::SUBTRACT) {
      alpha = -1.0f;
    }
    if (rhs_grad.get_float_ptr() != nullptr) {
      if (attrs.should_broadcast_rhs) {
        checkCUDNN(cudnnReduceTensor(handle.dnn,
                                     per_device_state.reduceAddDesc,
                                     nullptr /*indices*/,
                                     0 /*indicesSizeInBytes*/,
                                     handle.workSpace,
                                     handle.workSpaceSize,
                                     &alpha,
                                     per_device_state.outputTensor,
                                     output_grad.get_float_ptr(),
                                     &beta,
                                     per_device_state.inputRHSTensor,
                                     rhs_grad.get_float_ptr()));
      } else {
        checkCUDNN(cudnnAddTensor(handle.dnn,
                                  &alpha,
                                  per_device_state.outputTensor,
                                  output_grad.get_float_ptr(),
                                  &beta,
                                  per_device_state.inputRHSTensor,
                                  rhs_grad.get_float_ptr()));
      }
    }
  } else if (attrs.op == ElementBinaryOp::MULTIPLY) {
    float alpha1 = 1.0f, alpha2 = 1.0f, beta = 1.0f, zero = 0.0f;
    if (lhs_grad.get_float_ptr() != nullptr) {
      if (attrs.should_broadcast_lhs) {
        checkCUDNN(cudnnOpTensor(handle.dnn,
                                 per_device_state.opDesc,
                                 &alpha1,
                                 per_device_state.outputTensor,
                                 output_grad.get_float_ptr(),
                                 &alpha2,
                                 per_device_state.inputRHSTensor,
                                 rhs.get_float_ptr(),
                                 &zero,
                                 per_device_state.outputTensor,
                                 handle.workSpace));
        checkCUDNN(cudnnReduceTensor(
            handle.dnn,
            per_device_state.reduceAddDesc,
            nullptr /*indices*/,
            0 /*indicesSizeInBytes*/,
            (void *)((char *)handle.workSpace +
                     sizeof(*output_grad.get_float_ptr())),
            handle.workSpaceSize - sizeof(*output_grad.get_float_ptr()),
            &alpha1,
            per_device_state.outputTensor,
            handle.workSpace,
            &beta,
            per_device_state.inputLHSTensor,
            lhs_grad.get_float_ptr()));
      } else {
        checkCUDNN(cudnnOpTensor(handle.dnn,
                                 per_device_state.opDesc,
                                 &alpha1,
                                 per_device_state.outputTensor,
                                 output_grad.get_float_ptr(),
                                 &alpha2,
                                 per_device_state.inputRHSTensor,
                                 rhs.get_float_ptr(),
                                 &beta,
                                 per_device_state.inputLHSTensor,
                                 lhs_grad.get_float_ptr()));
      }
    }
    if (rhs_grad.get_float_ptr() != nullptr) {
      if (attrs.should_broadcast_rhs) {
        checkCUDNN(cudnnOpTensor(handle.dnn,
                                 per_device_state.opDesc,
                                 &alpha1,
                                 per_device_state.outputTensor,
                                 output_grad.get_float_ptr(),
                                 &alpha2,
                                 per_device_state.inputLHSTensor,
                                 lhs.get_float_ptr(),
                                 &zero,
                                 per_device_state.outputTensor,
                                 handle.workSpace));
        checkCUDNN(cudnnReduceTensor(
            handle.dnn,
            per_device_state.reduceAddDesc,
            nullptr /*indices*/,
            0 /*indicesSizeInBytes*/,
            (void *)((char *)handle.workSpace +
                     sizeof(*output_grad.get_float_ptr())),
            handle.workSpaceSize - sizeof(*output_grad.get_float_ptr()),
            &alpha1,
            per_device_state.outputTensor,
            handle.workSpace,
            &beta,
            per_device_state.inputRHSTensor,
            rhs_grad.get_float_ptr()));
      } else {
        checkCUDNN(cudnnOpTensor(handle.dnn,
                                 per_device_state.opDesc,
                                 &alpha1,
                                 per_device_state.outputTensor,
                                 output_grad.get_float_ptr(),
                                 &alpha2,
                                 per_device_state.inputLHSTensor,
                                 lhs.get_float_ptr(),
                                 &beta,
                                 per_device_state.inputRHSTensor,
                                 rhs_grad.get_float_ptr()));
      }
    }
  } else if (attrs.op == ElementBinaryOp::MIN ||
             attrs.op == ElementBinaryOp::MAX) {
    float alpha = 1.0f, beta = 1.0f;
    cudnnDataType_t dataType;
    int n;
    int dims[MAX_TENSOR_DIM];
    int strides[MAX_TENSOR_DIM];
    checkCUDNN(cudnnGetTensorNdDescriptor(per_device_state.outputTensor,
                                          MAX_TENSOR_DIM,
                                          &dataType,
                                          &n,
                                          dims,
                                          strides));
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
                                               attrs.op,
                                               output_grad.get_float_ptr(),
                                               lhs.get_float_ptr(),
                                               rhs.get_float_ptr(),
                                               lhs_grad.get_float_ptr(),
                                               rhs_grad.get_float_ptr());
  } else {
    assert(false && "Unsupported ElementWise Binary Type");
  }
}

void element_binary_gpu_cleanup_kernel(
    ElementBinaryPerDeviceState const &per_device_state) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow

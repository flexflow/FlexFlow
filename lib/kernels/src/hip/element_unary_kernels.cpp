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

#include "kernels/element_unary_kernels.h"
#include "device.h"
#include "kernels/datatype_dispatch.h"
#include "op-attrs/get_op_type.h"
#include <hip/hip_runtime.h>
#include <optional>

namespace FlexFlow {
namespace Kernels {
namespace ElementUnary {

using coord_t = long long;

static bool use_cudnn(OperatorType op_type) {
  switch (op_type) {
    case Op::RELU:
    case Op::SIGMOID:
    case Op::TANH:
    case Op::ELU:
      return true;
    default:
      return false;
  }
}

template <typename T>
T get_scalar(ElementUnaryUnifiedAttrs const &attrs) {
  if (std::holds_alternative<ElementScalarUnaryAttrs>(attrs)) {
    return (T)std::get<ElementScalarUnaryAttrs>(attrs).scalar;
  } else {
    T dummy_scalar;
    return dummy_scalar;
  }
}

ElementUnaryPerDeviceState init_kernel(ArrayShape const &input_shape,
                                       ArrayShape const &output_shape,
                                       ElementUnaryUnifiedAttrs const &attrs) {
  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t outputTensor;
  ffActivationDescriptor_t actiDesc;

  checkCUDNN(miopenCreateTensorDescriptor(&inputTensor));
  checkCUDNN(miopenCreateTensorDescriptor(&outputTensor));
  checkCUDNN(miopenCreateActivationDescriptor(&actiDesc));

  Op op_type = get_op_type(attrs);

  if (use_cudnn(op_type)) {
    miopenActivationMode_t mode;
    switch (op_type) {
      case Op::SIGMOID:
        mode = miopenActivationLOGISTIC;
        break;
      case Op::RELU:
        mode = miopenActivationRELU;
        break;
      case Op::TANH:
        mode = miopenActivationTANH;
        break;
      case Op::ELU:
        mode = miopenActivationELU;
        break;
      default:
        assert(false);
    }
  }
  checkCUDNN(miopenSetActivationDescriptor(actiDesc, mode, 0.0, 0.0, 0.0));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(inputTensor, input_shape));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(outputTensor, output_shape));
  return {inputTensor, outputTensor, actiDesc};
}

template <typename T>
__global__ void elewise_unary_forward_kernel(
    coord_t volume, const T scalar, OperatorType type, T const *in, T *out) {
  CUDA_KERNEL_LOOP(i, volume) {
    switch (type) {
      case OP_EXP: {
        out[i] = (T)exp((float)in[i]);
        break;
      }
      case OP_IDENTITY: {
        out[i] = in[i];
        break;
      }
      case OP_SCALAR_MULTIPLY: {
        out[i] = in[i] * scalar;
        break;
      }
      case OP_SCALAR_ADD: {
        out[i] = in[i] + scalar;
        break;
      }
      case OP_SCALAR_SUB: {
        out[i] = in[i] - scalar;
        break;
      }
      case OP_SCALAR_TRUE_DIV: {
        out[i] = in[i] / scalar;
        break;
      }
      case OP_GELU: {
        out[i] = (T)(in[i] * 0.5 * erfc(-in[i] * M_SQRT1_2));
        break;
      }
      case OP_RSQRT: {
        out[i] = (T)(1.0f / sqrt((float)in[i]));
        break;
      }
      case OP_POW: {
        out[i] = (T)(powf(in[i], scalar));
        break;
      }
      case OP_SIN: {
        out[i] = (T)sin((float)in[i]);
        break;
      }
      case OP_COS: {
        out[i] = (T)cos((float)in[i]);
        break;
      }
      default:
        assert(false);
    }
  }
}

template <typename T>
__global__ void elewise_unary_backward_kernel(coord_t volume,
                                              const T scalar,
                                              OperatorType type,
                                              T const *output,
                                              T const *output_grad,
                                              T const *input,
                                              T *input_grad) {
  CUDA_KERNEL_LOOP(i, volume) {
    switch (type) {
      case OP_EXP: {
        // TODO: change to use output instead of recomputing
        input_grad[i] += (T)(output_grad[i] * exp((float)input[i]));
        break;
      }
      case OP_IDENTITY: {
        input_grad[i] += output_grad[i];
        break;
      }
      case OP_SCALAR_MULTIPLY: {
        input_grad[i] += output_grad[i] * scalar;
        break;
      }
      case OP_SCALAR_ADD: {
        input_grad[i] += output_grad[i];
        break;
      }
      case OP_SCALAR_SUB: {
        input_grad[i] += output_grad[i];
        break;
      }
      case OP_SCALAR_TRUE_DIV: {
        input_grad[i] += output_grad[i] / scalar;
        break;
      }
      case OP_GELU: {
        input_grad[i] =
            (T)(output_grad[i] *
                (0.5 * erfc(-input[i] * M_SQRT1_2) -
                 0.5 * M_SQRT1_2 * input[i] * exp(-input[i] * input[i] * 0.5)));
        break;
      }
      case OP_RSQRT: {
        input_grad[i] =
            (T)(-0.5f * output_grad[i] * output[i] * output[i] * output[i]);
        break;
      }
      case OP_POW: {
        input_grad[i] =
            (T)(output_grad[i] * scalar * powf(input[i], scalar - 1));
        break;
      }
      case OP_SIN: {
        input_grad[i] += (T)(output_grad[i] * cos((float)input[i]));
        break;
      }
      case OP_COS: {
        input_grad[i] += (T)(output_grad[i] * -sin((float)input[i]));
        break;
      }
      default:
        assert(false);
    }
  }
}

template <DataType T>
struct ForwardKernel {
  void operator()(ffStream_t stream,
                  ElementUnaryPerDeviceState const &m,
                  ElementUnaryUnifiedAttrs const &attrs,
                  PerDeviceFFHandle const &handle,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) const {
    checkCUDNN(miopenSetStream(handle.dnn, stream));
    Op op_type = get_op_type(attrs);
    if (use_cudnn(op_type)) {
      float alpha = 1.0f, beta = 0.0f;
      checkCUDNN(miopenActivationForward(handle.dnn,
                                         m.actiDesc,
                                         &alpha,
                                         m.inputTensor,
                                         input.get<T>(),
                                         &beta,
                                         m.outputTensor,
                                         output.get<T>()));
    } else {
      size_t num_elements = input.shape.num_elements();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(elewise_unary_forward_kernel),
                         GET_BLOCKS(num_elements),
                         CUDA_NUM_THREADS,
                         0,
                         stream,
                         num_elements,
                         (T)m.scalar,
                         m.op_type,
                         input.get<T>(),
                         output.get<T>());
    }
  }
};

template <DataType T>
struct BackwardKernel {
  void operator()(ffStream_t stream,
                  ElementUnaryPerDeviceState const &m,
                  ElementUnaryUnifiedAttrs const &attrs,
                  PerDeviceFFHandle const &handle,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &input_grad,
                  GenericTensorAccessorR const &output,
                  GenericTensorAccessorR const &output_grad) {
    checkCUDNN(miopenSetStream(handle.dnn, stream));

    Op op_type = get_op_type(attrs);
    if (use_cudnn(op_type)) {
      float alpha = 1.0f;
      checkCUDNN(miopenActivationBackward(handle.dnn,
                                          m.actiDesc,
                                          &alpha,
                                          m.outputTensor,
                                          output.get<T>(),
                                          m.outputTensor,
                                          output_grad.get<T>()),
                 m.inputTensor,
                 input.get<T>(),
                 &beta,
                 m.inputTensor,
                 input_grad.get<T>());
    } else {
      size_t num_elements = input.shape.num_elements();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(elewise_unary_backward_kernel<T>),
                         GET_BLOCKS(num_elements),
                         CUDA_NUM_THREADS,
                         0,
                         stream,
                         num_elements,
                         m.scalar,
                         m.op_type,
                         output.get<T>(),
                         output_grad.get<T>(),
                         input.get<T>(),
                         input_grad.get<T>());
    }
  }
};
void forward_kernel(ffStream_t stream,
                    ElementUnaryPerDeviceState const &device_state,
                    ElementUnaryUnifiedAttrs const &attrs,
                    PerDeviceFFHandle const &handle,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output) {
  DataTypeDispatch1<ForwardKernel>{}(
      input.data_type, stream, device_state, attrs, handle, input, output);
}

void backward_kernel(ffStream_t stream,
                     ElementUnaryPerDeviceState const &device_state,
                     ElementUnaryUnifiedAttrs const &attrs,
                     PerDeviceFFHandle const &handle,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorW const &input_grad,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorR const &output_grad) {
  DataTypeDispatch1<BackwardKernel>{}(input.data_type,
                                      stream,
                                      device_state,
                                      attrs,
                                      handle,
                                      input,
                                      input_grad,
                                      output,
                                      output_grad);
}

} // namespace ElementUnary
} // namespace Kernels
} // namespace FlexFlow

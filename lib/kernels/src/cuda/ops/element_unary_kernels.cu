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
#include "kernels/datatype_dispatch.h"
#include "kernels/element_unary_kernels.h"
#include "op-attrs/get_op_type.h"
#include <optional>

namespace FlexFlow {
namespace Kernels {
namespace ElementUnary {

static bool use_cudnn(OperatorType op_type) {
  switch (op_type) {
    case OperatorType::RELU:
    case OperatorType::SIGMOID:
    case OperatorType::TANH:
    case OperatorType::ELU:
      return true;
    default:
      return false;
  }
}

static bool use_scalar(OperatorType op_type) {
  switch (op_type) {
    case OperatorType::SCALAR_MULTIPLY:
    case OperatorType::SCALAR_ADD:
    case OperatorType::SCALAR_SUB:
    case OperatorType::SCALAR_TRUE_DIV:
    case OperatorType::POW:
      return true;
    default:
      return false;
  }
}

static ElementUnaryPerDeviceState init_kernel(ArrayShape const &input_shape,
                                              ArrayShape const &output_shape,
                                              OperatorType op_type) {

  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t outputTensor;
  ffActivationDescriptor_t actiDesc;

  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));

  if (use_cudnn(op_type)) {
    cudnnActivationMode_t mode;
    switch (op_type) {
      case OperatorType::SIGMOID:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      case OperatorType::RELU:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case OperatorType::TANH:
        mode = CUDNN_ACTIVATION_TANH;
        break;
      case OperatorType::ELU:
        mode = CUDNN_ACTIVATION_ELU;
        break;
      default:
        assert(false);
    }
    checkCUDNN(
        cudnnSetActivationDescriptor(actiDesc, mode, CUDNN_PROPAGATE_NAN, 0.0));
    checkCUDNN(
        cudnnSetTensorDescriptorFromArrayShape(inputTensor, input_shape));
    checkCUDNN(
        cudnnSetTensorDescriptorFromArrayShape(outputTensor, output_shape));
  }

  return {inputTensor, outputTensor, actiDesc};
}

ElementUnaryPerDeviceState init_kernel(ArrayShape const &input_shape,
                                       ArrayShape const &output_shape,
                                       ElementUnaryAttrs const &attrs) {
  return init_kernel(input_shape, output_shape, get_op_type(attrs));
}

template <typename T>
__global__ void elewise_scalar_unary_forward_kernel(
    coord_t volume, T scalar, OperatorType type, T const *in, T *out) {
  CUDA_KERNEL_LOOP(i, volume) {
    switch (type) {
      case OperatorType::SCALAR_MULTIPLY: {
        out[i] = in[i] * scalar;
        break;
      }
      case OperatorType::SCALAR_ADD: {
        out[i] = in[i] + scalar;
        break;
      }
      case OperatorType::SCALAR_SUB: {
        out[i] = in[i] - scalar;
        break;
      }
      case OperatorType::SCALAR_TRUE_DIV: {
        out[i] = in[i] / scalar;
        break;
      }
      case OperatorType::POW: {
        out[i] = (T)(powf(in[i], scalar));
        break;
      }
      default:
        assert(false);
    }
  }
}

template <typename T>
__global__ void elewise_unary_forward_kernel(coord_t volume,
                                             OperatorType type,
                                             T const *in,
                                             T *out) {
  CUDA_KERNEL_LOOP(i, volume) {
    switch (type) {
      case OperatorType::EXP: {
        out[i] = (T)exp((float)in[i]);
        break;
      }
      case OperatorType::IDENTITY: {
        out[i] = in[i];
        break;
      }
      case OperatorType::GELU: {
        out[i] = (T)(in[i] * 0.5 * erfc(-in[i] * M_SQRT1_2));
        break;
      }
      case OperatorType::RSQRT: {
        out[i] = (T)(1.0f / sqrt((float)in[i]));
        break;
      }
      case OperatorType::SIN: {
        out[i] = (T)sin((float)in[i]);
        break;
      }
      case OperatorType::COS: {
        out[i] = (T)cos((float)in[i]);
        break;
      }
      default:
        assert(false);
    }
  }
}

template <typename T>
__global__ void elewise_scalar_unary_backward_kernel(coord_t volume,
                                                     T scalar,
                                                     OperatorType type,
                                                     T const *output,
                                                     T const *output_grad,
                                                     T const *input,
                                                     T *input_grad) {
  CUDA_KERNEL_LOOP(i, volume) {
    switch (type) {
      case OperatorType::SCALAR_MULTIPLY: {
        input_grad[i] += output_grad[i] * scalar;
        break;
      }
      case OperatorType::SCALAR_ADD: {
        input_grad[i] += output_grad[i];
        break;
      }
      case OperatorType::SCALAR_SUB: {
        input_grad[i] += output_grad[i];
        break;
      }
      case OperatorType::SCALAR_TRUE_DIV: {
        input_grad[i] += output_grad[i] / scalar;
        break;
      }
      case OperatorType::POW: {
        input_grad[i] =
            (T)(output_grad[i] * scalar * powf(input[i], scalar - 1));
        break;
      }
      default:
        assert(false);
    }
  }
}

template <typename T>
__global__ void elewise_unary_backward_kernel(coord_t volume,
                                              OperatorType type,
                                              T const *output,
                                              T const *output_grad,
                                              T const *input,
                                              T *input_grad) {
  CUDA_KERNEL_LOOP(i, volume) {
    switch (type) {
      case OperatorType::EXP: {
        // TODO: change to use output instead of recomputing
        input_grad[i] += (T)(output_grad[i] * exp((float)input[i]));
        break;
      }
      case OperatorType::IDENTITY: {
        input_grad[i] += output_grad[i];
        break;
      }
      case OperatorType::GELU: {
        input_grad[i] =
            (T)(output_grad[i] *
                (0.5 * erfc(-input[i] * M_SQRT1_2) -
                 0.5 * M_SQRT1_2 * input[i] * exp(-input[i] * input[i] * 0.5)));
        break;
      }
      case OperatorType::RSQRT: {
        input_grad[i] =
            (T)(-0.5f * output_grad[i] * output[i] * output[i] * output[i]);
        break;
      }
      case OperatorType::SIN: {
        input_grad[i] += (T)(output_grad[i] * cos((float)input[i]));
        break;
      }
      case OperatorType::COS: {
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
                  OperatorType op_type,
                  std::optional<float> scalar,
                  PerDeviceFFHandle const &handle,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) const {
    checkCUDNN(cudnnSetStream(handle.dnn, stream));
    if (use_cudnn(op_type)) {
      float alpha = 1.0f, beta = 0.0f;
      checkCUDNN(cudnnActivationForward(handle.dnn,
                                        m.actiDesc,
                                        &alpha,
                                        m.inputTensor,
                                        input.get<T>(),
                                        &beta,
                                        m.outputTensor,
                                        output.get<T>()));
    } else if (use_scalar(op_type)) {
      assert(scalar.has_value());
      size_t num_elements = input.shape.num_elements();
      elewise_scalar_unary_forward_kernel<real_type_t<T>>
          <<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS, 0, stream>>>(
              num_elements,
              static_cast<real_type_t<T>>(scalar.value()),
              op_type,
              input.get<T>(),
              output.get<T>());
    } else {
      size_t num_elements = input.shape.num_elements();
      elewise_unary_forward_kernel<real_type_t<T>>
          <<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS, 0, stream>>>(
              num_elements, op_type, input.get<T>(), output.get<T>());
    }
  }
};

template <DataType T>
struct BackwardKernel {
  void operator()(ffStream_t stream,
                  ElementUnaryPerDeviceState const &m,
                  OperatorType op_type,
                  std::optional<float> scalar,
                  PerDeviceFFHandle const &handle,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &input_grad,
                  GenericTensorAccessorR const &output,
                  GenericTensorAccessorR const &output_grad) {
    checkCUDNN(cudnnSetStream(handle.dnn, stream));

    if (use_cudnn(op_type)) {
      float alpha = 1.0f;
      checkCUDNN(cudnnActivationBackward(handle.dnn,
                                         m.actiDesc,
                                         &alpha,
                                         m.outputTensor,
                                         output.get<T>(),
                                         m.outputTensor,
                                         output_grad.get<T>(),
                                         m.inputTensor,
                                         input.get<T>(),
                                         &alpha,
                                         m.inputTensor,
                                         input_grad.get<T>()));
    } else if (use_scalar(op_type)) {
      assert(scalar.has_value());
      size_t num_elements = input.shape.num_elements();
      elewise_scalar_unary_backward_kernel<real_type_t<T>>
          <<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS, 0, stream>>>(
              num_elements,
              static_cast<real_type_t<T>>(scalar.value()),
              op_type,
              output.get<T>(),
              output_grad.get<T>(),
              input.get<T>(),
              input_grad.get<T>());
    } else {
      size_t num_elements = input.shape.num_elements();
      elewise_unary_backward_kernel<real_type_t<T>>
          <<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS, 0, stream>>>(
              num_elements,
              op_type,
              output.get<T>(),
              output_grad.get<T>(),
              input.get<T>(),
              input_grad.get<T>());
    }
  }
};

void forward_kernel(ffStream_t stream,
                    ElementUnaryPerDeviceState const &device_state,
                    ElementUnaryAttrs const &attrs,
                    PerDeviceFFHandle const &handle,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output) {
  DataTypeDispatch1<ForwardKernel>{}(input.data_type,
                                     stream,
                                     device_state,
                                     get_op_type(attrs),
                                     attrs.scalar,
                                     handle,
                                     input,
                                     output);
}

void backward_kernel(ffStream_t stream,
                     ElementUnaryPerDeviceState const &device_state,
                     ElementUnaryAttrs const &attrs,
                     PerDeviceFFHandle const &handle,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorW const &input_grad,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorR const &output_grad) {
  DataTypeDispatch1<BackwardKernel>{}(input.data_type,
                                      stream,
                                      device_state,
                                      get_op_type(attrs),
                                      attrs.scalar,
                                      handle,
                                      input,
                                      input_grad,
                                      output,
                                      output_grad);
}

} // namespace ElementUnary
} // namespace Kernels
} // namespace FlexFlow

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

#include "flexflow/ops/element_unary.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Domain;

/*static*/
void ElementUnary::init_kernel(ElementUnaryMeta *m,
                               Domain const &input_domain,
                               Domain const &output_domain) {
  miopenActivationMode_t mode;
  switch (m->op_type) {
    case OP_SIGMOID:
      mode = miopenActivationLOGISTIC;
      break;
    case OP_RELU:
      mode = miopenActivationRELU;
      break;
    case OP_TANH:
      mode = miopenActivationTANH;
      break;
    case OP_ELU:
      mode = miopenActivationELU;
      break;
    default:
      assert(false);
  }
  checkCUDNN(miopenSetActivationDescriptor(m->actiDesc, mode, 0.0, 0.0, 0.0));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(
      m->inputTensor, input_domain, m->data_type));
  // input_domain == output_domain
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(
      m->outputTensor, output_domain, m->data_type));
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
        out[i] = (T)(in[i] * static_cast<T>(0.5f) *
                     static_cast<T>(erfc(static_cast<float>(
                         -in[i] * static_cast<T>(M_SQRT1_2)))));
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

/*static*/
template <typename T>
void ElementUnary::forward_kernel(ElementUnaryMeta const *m,
                                  T const *input_ptr,
                                  T *output_ptr,
                                  size_t num_elements,
                                  hipStream_t stream) {
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

  if (use_cudnn(m->op_type)) {
    float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(miopenActivationForward(m->handle.dnn,
                                       m->actiDesc,
                                       &alpha,
                                       m->inputTensor,
                                       input_ptr,
                                       &beta,
                                       m->outputTensor,
                                       output_ptr));
  } else {
    hipLaunchKernelGGL(elewise_unary_forward_kernel,
                       GET_BLOCKS(num_elements),
                       CUDA_NUM_THREADS,
                       0,
                       stream,
                       num_elements,
                       (T)m->scalar,
                       m->op_type,
                       input_ptr,
                       output_ptr);
  }
}

/*static*/
template <typename T>
void ElementUnary::forward_kernel_wrapper(ElementUnaryMeta const *m,
                                          T const *input_ptr,
                                          T *output_ptr,
                                          size_t num_elements) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  ElementUnary::forward_kernel<T>(
      m, input_ptr, output_ptr, num_elements, stream);
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
                (0.5 * static_cast<T>(erfc(-input[i] * M_SQRT1_2)) +
                 0.5 * M_SQRT1_2 * input[i] *
                     ((2 / sqrt(M_PI)) * exp(-input[i] * input[i] * 0.5))));
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

/*static*/
template <typename T>
void ElementUnary::backward_kernel(ElementUnaryMeta const *m,
                                   T const *input_ptr,
                                   T *input_grad_ptr,
                                   T const *output_ptr,
                                   T const *output_grad_ptr,
                                   size_t num_elements,
                                   hipStream_t stream) {
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

  if (use_cudnn(m->op_type)) {
    float alpha = 1.0f;
    float beta = 0.0f;
    checkCUDNN(miopenActivationBackward(m->handle.dnn,
                                        m->actiDesc,
                                        &alpha,
                                        m->outputTensor,
                                        output_ptr,
                                        m->outputTensor,
                                        output_grad_ptr,
                                        m->inputTensor,
                                        input_ptr,
                                        &beta,
                                        m->inputTensor,
                                        input_grad_ptr));
  } else {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(elewise_unary_backward_kernel<T>),
                       GET_BLOCKS(num_elements),
                       CUDA_NUM_THREADS,
                       0,
                       stream,
                       num_elements,
                       m->scalar,
                       m->op_type,
                       output_ptr,
                       output_grad_ptr,
                       input_ptr,
                       input_grad_ptr);
  }
}

/*static*/
template <typename T>
void ElementUnary::backward_kernel_wrapper(ElementUnaryMeta const *m,
                                           T const *input_ptr,
                                           T *input_grad_ptr,
                                           T const *output_ptr,
                                           T const *output_grad_ptr,
                                           size_t num_elements) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  ElementUnary::backward_kernel<T>(m,
                                   input_ptr,
                                   input_grad_ptr,
                                   output_ptr,
                                   output_grad_ptr,
                                   num_elements,
                                   stream);
}

ElementUnaryMeta::ElementUnaryMeta(FFHandler handler, ElementUnary const *unary)
    : OpMeta(handler, unary) {
  checkCUDNN(miopenCreateTensorDescriptor(&inputTensor));
  checkCUDNN(miopenCreateTensorDescriptor(&outputTensor));
  checkCUDNN(miopenCreateActivationDescriptor(&actiDesc));
}

template void
    ElementUnary::forward_kernel_wrapper<half>(ElementUnaryMeta const *m,
                                               half const *input_ptr,
                                               half *output_ptr,
                                               size_t num_elements);
template void
    ElementUnary::forward_kernel_wrapper<float>(ElementUnaryMeta const *m,
                                                float const *input_ptr,
                                                float *output_ptr,
                                                size_t num_elements);
template void
    ElementUnary::forward_kernel_wrapper<double>(ElementUnaryMeta const *m,
                                                 double const *input_ptr,
                                                 double *output_ptr,
                                                 size_t num_elements);
template void
    ElementUnary::forward_kernel_wrapper<int32_t>(ElementUnaryMeta const *m,
                                                  int32_t const *input_ptr,
                                                  int32_t *output_ptr,
                                                  size_t num_elements);
template void
    ElementUnary::forward_kernel_wrapper<int64_t>(ElementUnaryMeta const *m,
                                                  int64_t const *input_ptr,
                                                  int64_t *output_ptr,
                                                  size_t num_elements);

template void
    ElementUnary::backward_kernel_wrapper<float>(ElementUnaryMeta const *m,
                                                 float const *input_ptr,
                                                 float *input_grad_ptr,
                                                 float const *output_ptr,
                                                 float const *output_grad_ptr,
                                                 size_t num_elements);
template void
    ElementUnary::backward_kernel_wrapper<double>(ElementUnaryMeta const *m,
                                                  double const *input_ptr,
                                                  double *input_grad_ptr,
                                                  double const *output_ptr,
                                                  double const *output_grad_ptr,
                                                  size_t num_elements);
template void ElementUnary::backward_kernel_wrapper<int32_t>(
    ElementUnaryMeta const *m,
    int32_t const *input_ptr,
    int32_t *input_grad_ptr,
    int32_t const *output_ptr,
    int32_t const *output_grad_ptr,
    size_t num_elements);
template void ElementUnary::backward_kernel_wrapper<int64_t>(
    ElementUnaryMeta const *m,
    int64_t const *input_ptr,
    int64_t *input_grad_ptr,
    int64_t const *output_ptr,
    int64_t const *output_grad_ptr,
    size_t num_elements);

}; // namespace FlexFlow

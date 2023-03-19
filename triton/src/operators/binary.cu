/* Copyright 2022 NVIDIA CORPORATION
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

#include "binary.h"

#include "mathtypes/half.h"

using namespace Legion;

namespace triton {
namespace backend {
namespace legion {

__global__ static void binary_forward_half(__half const *input0,
                                           __half const *input1,
                                           __half *output,
                                           const __half alpha,
                                           const __half beta,
                                           const OperatorType optype,
                                           const size_t volume) {
  const size_t offset = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (offset >= volume) {
    return;
  }
  switch (optype) {
    case OP_EW_ADD: {
      output[offset] =
          alpha * (input0[offset] + input1[offset]) + beta * output[offset];
      break;
    }
    case OP_EW_SUB: {
      output[offset] =
          alpha * (input0[offset] - input1[offset]) + beta * output[offset];
      break;
    }
    case OP_EW_MUL: {
      output[offset] =
          alpha * (input0[offset] * input1[offset]) + beta * output[offset];
      break;
    }
    case OP_EW_DIV: {
      output[offset] =
          alpha * (input0[offset] / input1[offset]) + beta * output[offset];
      break;
    }
    default:
      break;
  }
}

__global__ static void binary_forward_float(float const *input0,
                                            float const *input1,
                                            float *output,
                                            float const alpha,
                                            float const beta,
                                            const OperatorType optype,
                                            const size_t volume) {
  const size_t offset = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (offset >= volume) {
    return;
  }
  switch (optype) {
    case OP_EW_ADD: {
      output[offset] =
          alpha * (input0[offset] + input1[offset]) + beta * output[offset];
      break;
    }
    case OP_EW_SUB: {
      output[offset] =
          alpha * (input0[offset] - input1[offset]) + beta * output[offset];
      break;
    }
    case OP_EW_MUL: {
      output[offset] =
          alpha * (input0[offset] * input1[offset]) + beta * output[offset];
      break;
    }
    case OP_EW_DIV: {
      output[offset] =
          alpha * (input0[offset] / input1[offset]) + beta * output[offset];
      break;
    }
    default:
      break;
  }
}

__global__ static void binary_forward_double(double const *input0,
                                             double const *input1,
                                             double *output,
                                             double const alpha,
                                             double const beta,
                                             const OperatorType optype,
                                             const size_t volume) {
  const size_t offset = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (offset >= volume) {
    return;
  }
  switch (optype) {
    case OP_EW_ADD: {
      output[offset] =
          alpha * (input0[offset] + input1[offset]) + beta * output[offset];
      break;
    }
    case OP_EW_SUB: {
      output[offset] =
          alpha * (input0[offset] - input1[offset]) + beta * output[offset];
      break;
    }
    case OP_EW_MUL: {
      output[offset] =
          alpha * (input0[offset] * input1[offset]) + beta * output[offset];
      break;
    }
    case OP_EW_DIV: {
      output[offset] =
          alpha * (input0[offset] / input1[offset]) + beta * output[offset];
      break;
    }
    default:
      break;
  }
}

__global__ static void binary_forward_int8(int8_t const *input0,
                                           int8_t const *input1,
                                           int8_t *output,
                                           const int8_t alpha,
                                           const int8_t beta,
                                           const OperatorType optype,
                                           const size_t volume) {
  const size_t offset = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (offset >= volume) {
    return;
  }
  switch (optype) {
    case OP_EW_ADD: {
      output[offset] =
          alpha * (input0[offset] + input1[offset]) + beta * output[offset];
      break;
    }
    case OP_EW_SUB: {
      output[offset] =
          alpha * (input0[offset] - input1[offset]) + beta * output[offset];
      break;
    }
    case OP_EW_MUL: {
      output[offset] =
          alpha * (input0[offset] * input1[offset]) + beta * output[offset];
      break;
    }
    case OP_EW_DIV: {
      output[offset] =
          alpha * (input0[offset] / input1[offset]) + beta * output[offset];
      break;
    }
    default:
      break;
  }
}

__host__
    /*static*/ void
    BinaryOperator::forward_kernel(BinaryArgs const *args,
                                   ::cudaStream_t stream,
                                   void const *input0_ptr,
                                   void const *input1_ptr,
                                   void *output_ptr,
                                   size_t num_elements) {
  if (use_cudnn(args->op_type, args->datatype)) {
    switch (args->datatype) {
      case DataType::DT_DOUBLE: {
        double alpha0 = 1.0,
               alpha1 = (args->op_type == OperatorType::OP_EW_SUB) ? -1.0 : 1.0,
               beta = 0.0;
        CHECK_CUDNN(cudnnOpTensor(args->cudnn,
                                  args->opDesc,
                                  &alpha0,
                                  args->input0Tensor,
                                  input0_ptr,
                                  &alpha1,
                                  args->input1Tensor,
                                  input1_ptr,
                                  &beta,
                                  args->outputTensor,
                                  output_ptr));
        break;
      }
      case DataType::DT_FLOAT: {
        float alpha0 = 1.f,
              alpha1 = (args->op_type == OperatorType::OP_EW_SUB) ? -1.f : 1.f,
              beta = 0.f;
        CHECK_CUDNN(cudnnOpTensor(args->cudnn,
                                  args->opDesc,
                                  &alpha0,
                                  args->input0Tensor,
                                  input0_ptr,
                                  &alpha1,
                                  args->input1Tensor,
                                  input1_ptr,
                                  &beta,
                                  args->outputTensor,
                                  output_ptr));
        break;
      }
      case DataType::DT_INT8: {
        int8_t alpha0 = 1,
               alpha1 = (args->op_type == OperatorType::OP_EW_SUB) ? -1 : 1,
               beta = 0;
        CHECK_CUDNN(cudnnOpTensor(args->cudnn,
                                  args->opDesc,
                                  &alpha0,
                                  args->input0Tensor,
                                  input0_ptr,
                                  &alpha1,
                                  args->input1Tensor,
                                  input1_ptr,
                                  &beta,
                                  args->outputTensor,
                                  output_ptr));
        break;
      }
      case DataType::DT_HALF: {
        __half alpha0 = 1.f,
               alpha1 = (args->op_type == OperatorType::OP_EW_SUB) ? -1.f : 1.f,
               beta = 0.f;
        CHECK_CUDNN(cudnnOpTensor(args->cudnn,
                                  args->opDesc,
                                  &alpha0,
                                  args->input0Tensor,
                                  input0_ptr,
                                  &alpha1,
                                  args->input1Tensor,
                                  input1_ptr,
                                  &beta,
                                  args->outputTensor,
                                  output_ptr));
        break;
      }
      default:
        abort();
        break;
    }
  } else {
    const size_t blocks =
        (num_elements + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;
    assert((args->op_type == OP_EW_ADD) || (args->op_type == OP_EW_SUB) ||
           (args->op_type == OP_EW_MUL) || (args->op_type == OP_EW_DIV));
    switch (args->datatype) {
      case DataType::DT_DOUBLE: {
        double alpha = 1.0, beta = 0.0;
        binary_forward_double<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            (double const *)input0_ptr,
            (double const *)input1_ptr,
            (double *)output_ptr,
            alpha,
            beta,
            args->op_type,
            num_elements);
        break;
      }
      case DataType::DT_FLOAT: {
        float alpha = 1.f, beta = 0.f;
        binary_forward_float<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            (float const *)input0_ptr,
            (float const *)input1_ptr,
            (float *)output_ptr,
            alpha,
            beta,
            args->op_type,
            num_elements);
        break;
      }
      case DataType::DT_INT8: {
        int8_t alpha = 1, beta = 0;
        binary_forward_int8<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            (int8_t const *)input0_ptr,
            (int8_t const *)input1_ptr,
            (int8_t *)output_ptr,
            alpha,
            beta,
            args->op_type,
            num_elements);
        break;
      }
      case DataType::DT_HALF: {
        __half alpha = 1.f, beta = 0.f;
        binary_forward_half<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            (__half const *)input0_ptr,
            (__half const *)input1_ptr,
            (__half *)output_ptr,
            alpha,
            beta,
            args->op_type,
            num_elements);
        break;
      }
      default:
        abort();
        break;
    }
  }
}

} // namespace legion
} // namespace backend
} // namespace triton

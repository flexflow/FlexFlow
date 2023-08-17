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

#include "unary.h"

#include "mathtypes/half.h"

using namespace Legion;

namespace triton { namespace backend { namespace legion {

template <typename TI, typename TO>
__global__ static void
gpu_forward_cast(const TI* input, TO* output, const size_t volume)
{
  const size_t offset = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (offset >= volume)
    return;
  output[offset] = (TO)input[offset];
}

// Some unfortunate specializations because the compiler can't figure
// out the best intermedidate type to convert half types to
template <>
__global__ void
gpu_forward_cast<__half, int8_t>(
    const __half* input, int8_t* output, const size_t volume)
{
  const size_t offset = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (offset >= volume)
    return;
  output[offset] = (short)input[offset];
}

template <>
__global__ void
gpu_forward_cast<__half, int64_t>(
    const __half* input, int64_t* output, const size_t volume)
{
  const size_t offset = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (offset >= volume)
    return;
  output[offset] = (long long)input[offset];
}

template <>
__global__ void
gpu_forward_cast<int64_t, __half>(
    const int64_t* input, __half* output, const size_t volume)
{
  const size_t offset = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (offset >= volume)
    return;
  output[offset] = (long long)input[offset];
}

template <>
__global__ void
gpu_forward_cast<__half, uint8_t>(
    const __half* input, uint8_t* output, const size_t volume)
{
  const size_t offset = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (offset >= volume)
    return;
  output[offset] = (unsigned short)input[offset];
}

template <>
__global__ void
gpu_forward_cast<__half, uint64_t>(
    const __half* input, uint64_t* output, const size_t volume)
{
  const size_t offset = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (offset >= volume)
    return;
  output[offset] = (unsigned long long)input[offset];
}

template <>
__global__ void
gpu_forward_cast<uint64_t, __half>(
    const uint64_t* input, __half* output, const size_t volume)
{
  const size_t offset = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (offset >= volume)
    return;
  output[offset] = (unsigned long long)input[offset];
}

__global__ static void
unary_forward_half(
    const __half* input, __half* output, const __half alpha, const __half beta,
    const __half scalar, const OperatorType optype, const size_t volume)
{
  const size_t offset = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (offset >= volume)
    return;
  switch (optype) {
    case OP_EXP: {
      output[offset] = alpha * hexp(input[offset]) + beta * output[offset];
      break;
    }
    case OP_LOG: {
      output[offset] = alpha * hlog(input[offset]) + beta * output[offset];
      break;
    }
    case OP_SQRT: {
      output[offset] = alpha * hsqrt(input[offset]) + beta * output[offset];
      break;
    }
    case OP_IDENTITY: {
      output[offset] = input[offset];
      break;
    }
    case OP_SCALAR_MULTIPLY: {
      output[offset] = input[offset] * scalar;
      break;
    }
    case OP_SCALAR_ADD: {
      output[offset] = input[offset] + scalar;
      break;
    }
    case OP_SCALAR_SUB: {
      output[offset] = input[offset] - scalar;
      break;
    }
    case OP_SCALAR_TRUE_DIV: {
      output[offset] = input[offset] / scalar;
      break;
    }
    case OP_GELU: {
      output[offset] = __hmul(
          __hmul(input[offset], 0.5f),
          erfcf(__hmul(-input[offset], M_SQRT1_2)));
      break;
    }
    case OP_RECIPROCAL: {
      output[offset] = __hdiv(__half(1.f), input[offset]);
      break;
    }
    default:
      break;
  }
}

__global__ static void
unary_forward_float(
    const float* input, float* output, const float alpha, const float beta,
    const float scalar, const OperatorType optype, const size_t volume)
{
  const size_t offset = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (offset >= volume)
    return;
  switch (optype) {
    case OP_EXP: {
      output[offset] = alpha * expf(input[offset]) + beta * output[offset];
      break;
    }
    case OP_LOG: {
      output[offset] = alpha * logf(input[offset]) + beta * output[offset];
      break;
    }
    case OP_SQRT: {
      output[offset] = alpha * sqrtf(input[offset]) + beta * output[offset];
      break;
    }
    case OP_IDENTITY: {
      output[offset] = input[offset];
      break;
    }
    case OP_SCALAR_MULTIPLY: {
      output[offset] = input[offset] * scalar;
      break;
    }
    case OP_SCALAR_ADD: {
      output[offset] = input[offset] + scalar;
      break;
    }
    case OP_SCALAR_SUB: {
      output[offset] = input[offset] - scalar;
      break;
    }
    case OP_SCALAR_TRUE_DIV: {
      output[offset] = input[offset] / scalar;
      break;
    }
    case OP_GELU: {
      output[offset] = input[offset] * 0.5f * erfc(-input[offset] * M_SQRT1_2);
      break;
    }
    case OP_RECIPROCAL: {
      output[offset] = 1.f / input[offset];
      break;
    }
    default:
      break;
  }
}

__global__ static void
unary_forward_double(
    const double* input, double* output, const double alpha, const double beta,
    const double scalar, const OperatorType optype, const size_t volume)
{
  const size_t offset = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (offset >= volume)
    return;
  switch (optype) {
    case OP_EXP: {
      output[offset] = alpha * exp(input[offset]) + beta * output[offset];
      break;
    }
    case OP_LOG: {
      output[offset] = alpha * log(input[offset]) + beta * output[offset];
      break;
    }
    case OP_SQRT: {
      output[offset] = alpha * sqrt(input[offset]) + beta * output[offset];
      break;
    }
    case OP_IDENTITY: {
      output[offset] = input[offset];
      break;
    }
    case OP_SCALAR_MULTIPLY: {
      output[offset] = input[offset] * scalar;
      break;
    }
    case OP_SCALAR_ADD: {
      output[offset] = input[offset] + scalar;
      break;
    }
    case OP_SCALAR_SUB: {
      output[offset] = input[offset] - scalar;
      break;
    }
    case OP_SCALAR_TRUE_DIV: {
      output[offset] = input[offset] / scalar;
      break;
    }
    case OP_GELU: {
      output[offset] = input[offset] * 0.5 * erfc(-input[offset] * M_SQRT1_2);
      break;
    }
    case OP_RECIPROCAL: {
      output[offset] = 1.0 / input[offset];
      break;
    }
    default:
      break;
  }
}

template <typename T>
__host__ static void
forward_cast(
    DataType output_type, ::cudaStream_t stream, const void* input_ptr,
    void* output_ptr, size_t num_elements)
{
  const size_t blocks =
      (num_elements + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;
  switch (output_type) {
    case DT_HALF: {
      gpu_forward_cast<T, __half><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          (const T*)input_ptr, (__half*)output_ptr, num_elements);
      break;
    }
    case DT_FLOAT: {
      gpu_forward_cast<T, float><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          (const T*)input_ptr, (float*)output_ptr, num_elements);
      break;
    }
    case DT_DOUBLE: {
      gpu_forward_cast<T, double><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          (const T*)input_ptr, (double*)output_ptr, num_elements);
      break;
    }
    case DT_INT8: {
      gpu_forward_cast<T, int8_t><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          (const T*)input_ptr, (int8_t*)output_ptr, num_elements);
      break;
    }
    case DT_INT16: {
      gpu_forward_cast<T, int16_t><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          (const T*)input_ptr, (int16_t*)output_ptr, num_elements);
      break;
    }
    case DT_INT32: {
      gpu_forward_cast<T, int32_t><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          (const T*)input_ptr, (int32_t*)output_ptr, num_elements);
      break;
    }
    case DT_INT64: {
      gpu_forward_cast<T, int64_t><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          (const T*)input_ptr, (int64_t*)output_ptr, num_elements);
      break;
    }
    case DT_UINT8: {
      gpu_forward_cast<T, uint8_t><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          (const T*)input_ptr, (uint8_t*)output_ptr, num_elements);
      break;
    }
    case DT_UINT16: {
      gpu_forward_cast<T, uint16_t><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          (const T*)input_ptr, (uint16_t*)output_ptr, num_elements);
      break;
    }
    case DT_UINT32: {
      gpu_forward_cast<T, uint32_t><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          (const T*)input_ptr, (uint32_t*)output_ptr, num_elements);
      break;
    }
    case DT_UINT64: {
      gpu_forward_cast<T, uint64_t><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          (const T*)input_ptr, (uint64_t*)output_ptr, num_elements);
      break;
    }
    case DT_BOOLEAN: {
      gpu_forward_cast<T, bool><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          (const T*)input_ptr, (bool*)output_ptr, num_elements);
      break;
    }
    default:
      abort();
  }
}

__host__
    /*static*/ void
    UnaryOperator::forward_kernel(
        const UnaryArgs* args, ::cudaStream_t stream, const void* input_ptr,
        void* output_ptr, size_t num_elements)
{
  if (args->op_type == OP_CAST) {
    switch (args->datatype) {
      case DT_HALF: {
        forward_cast<__half>(
            args->casttype, stream, input_ptr, output_ptr, num_elements);
        break;
      }
      case DT_FLOAT: {
        forward_cast<float>(
            args->casttype, stream, input_ptr, output_ptr, num_elements);
        break;
      }
      case DT_DOUBLE: {
        forward_cast<double>(
            args->casttype, stream, input_ptr, output_ptr, num_elements);
        break;
      }
      case DT_INT8: {
        forward_cast<int8_t>(
            args->casttype, stream, input_ptr, output_ptr, num_elements);
        break;
      }
      case DT_INT16: {
        forward_cast<int16_t>(
            args->casttype, stream, input_ptr, output_ptr, num_elements);
        break;
      }
      case DT_INT32: {
        forward_cast<int32_t>(
            args->casttype, stream, input_ptr, output_ptr, num_elements);
        break;
      }
      case DT_INT64: {
        forward_cast<int64_t>(
            args->casttype, stream, input_ptr, output_ptr, num_elements);
        break;
      }
      case DT_UINT8: {
        forward_cast<uint8_t>(
            args->casttype, stream, input_ptr, output_ptr, num_elements);
        break;
      }
      case DT_UINT16: {
        forward_cast<uint16_t>(
            args->casttype, stream, input_ptr, output_ptr, num_elements);
        break;
      }
      case DT_UINT32: {
        forward_cast<uint32_t>(
            args->casttype, stream, input_ptr, output_ptr, num_elements);
        break;
      }
      case DT_UINT64: {
        forward_cast<uint64_t>(
            args->casttype, stream, input_ptr, output_ptr, num_elements);
        break;
      }
      case DT_BOOLEAN: {
        forward_cast<bool>(
            args->casttype, stream, input_ptr, output_ptr, num_elements);
        break;
      }
      default:
        abort();
    }
  } else if (use_cudnn(args->op_type)) {
    if (args->datatype == DT_DOUBLE) {
      double alpha = 1.0, beta = 0.0;
      CHECK_CUDNN(cudnnActivationForward(
          args->cudnn, args->actiDesc, &alpha, args->inputTensor, input_ptr,
          &beta, args->outputTensor, output_ptr));
    } else {
      float alpha = 1.f, beta = 0.f;
      CHECK_CUDNN(cudnnActivationForward(
          args->cudnn, args->actiDesc, &alpha, args->inputTensor, input_ptr,
          &beta, args->outputTensor, output_ptr));
    }
  } else {
    const size_t blocks =
        (num_elements + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;
    assert(
        (args->op_type == OP_EXP) || (args->op_type == OP_LOG) ||
        (args->op_type == OP_SQRT) || (args->op_type == OP_IDENTITY) ||
        (args->op_type == OP_SCALAR_MULTIPLY) ||
        (args->op_type == OP_SCALAR_ADD) || (args->op_type == OP_SCALAR_SUB) ||
        (args->op_type == OP_SCALAR_TRUE_DIV) || (args->op_type == OP_GELU) ||
        (OP_RECIPROCAL));
    switch (args->datatype) {
      case DT_HALF: {
        __half alpha = 1.f, beta = 0.f;
        unary_forward_half<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            (const __half*)input_ptr, (__half*)output_ptr, alpha, beta,
            args->scalar.half_value, args->op_type, num_elements);
        break;
      }
      case DT_FLOAT: {
        float alpha = 1.f, beta = 0.f;
        unary_forward_float<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            (const float*)input_ptr, (float*)output_ptr, alpha, beta,
            args->scalar.float_value, args->op_type, num_elements);
        break;
      }
      case DT_DOUBLE: {
        double alpha = 1.0, beta = 0.0;
        unary_forward_double<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            (const double*)input_ptr, (double*)output_ptr, alpha, beta,
            args->scalar.double_value, args->op_type, num_elements);
        break;
      }
      default:
        // TODO support for other data types like int8
        abort();
    }
  }
}

}}}  // namespace triton::backend::legion

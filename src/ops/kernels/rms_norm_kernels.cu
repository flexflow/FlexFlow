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

#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/kernels/rms_norm_kernels.h"
#include "flexflow/ops/rms_norm.h"
#include "flexflow/utils/cuda_helper.h"
#include <cublas_v2.h>

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;

#define C10_WARP_SIZE 32
constexpr int kCUDABlockReduceNumThreads = 512;
constexpr int kCUDANumThreads = 256;

RMSNormMeta::RMSNormMeta(FFHandler handler,
                         RMSNorm const *rms,
                         MemoryAllocator &gpu_mem_allocator)
    : OpMeta(handler, rms) {
  eps = rms->eps;
  alpha = 1.0f;
  beta = 0.0f;

  in_dim = rms->data_dim;
  batch_size = rms->effective_batch_size;
  num_elements = in_dim * batch_size;

  DataType data_type = rms->weights[0]->data_type;
  size_t rms_ptr_size = batch_size;
  size_t c2_ptr_size = rms_ptr_size;
  size_t norm_ptr_size = num_elements;
  size_t totalSize =
      (rms_ptr_size + c2_ptr_size + norm_ptr_size) * data_type_size(data_type);
  gpu_mem_allocator.create_legion_instance(reserveInst, totalSize);
  rms_ptr = gpu_mem_allocator.allocate_instance_untyped(
      rms_ptr_size * data_type_size(data_type));
  c2_ptr = gpu_mem_allocator.allocate_instance_untyped(
      c2_ptr_size * data_type_size(data_type));
  norm_ptr = gpu_mem_allocator.allocate_instance_untyped(
      norm_ptr_size * data_type_size(data_type));
}
RMSNormMeta::~RMSNormMeta(void) {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
}

namespace Kernels {
namespace RMSNorm {

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value,
                                            unsigned int delta,
                                            int width = warpSize,
                                            unsigned int mask = 0xffffffff) {
#ifndef __HIP_PLATFORM_HCC__
  return __shfl_down_sync(mask, value, delta, width);
#else
  return __shfl_down(value, delta, width);
#endif
}

template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

template <typename T>
__inline__ __device__ T BlockReduceSum(T val, T *shared) {
  int const lid = threadIdx.x % C10_WARP_SIZE;
  int const wid = threadIdx.x / C10_WARP_SIZE;
  val = WarpReduceSum(val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (threadIdx.x < (blockDim.x / C10_WARP_SIZE)) ? shared[lid] : T(0);
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}

template <typename T>
__inline__ __device__ T BlockReduceSum(T val, T *shared, int max_num_threads) {
  int const lid = threadIdx.x % C10_WARP_SIZE;
  int const wid = threadIdx.x / C10_WARP_SIZE;
  val = WarpReduceSum(val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (threadIdx.x < (min(blockDim.x, max_num_threads) / C10_WARP_SIZE))
            ? shared[lid]
            : T(0);
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}

template <typename T>
__global__ void RMSNormFusedForwardKernel(int64_t N,
                                          float eps,
                                          T const *X,
                                          T *rms,
                                          T *Y,
                                          T const *weights,
                                          T *output) {
  __shared__ float v_shared[C10_WARP_SIZE];
  int64_t const i = blockIdx.x;
  float sum = 0.0f;
  for (int64_t j = threadIdx.x; j < N;
       j += min(blockDim.x, kCUDABlockReduceNumThreads)) {
    int64_t const index = i * N + j;
    sum += (static_cast<float>(X[index]) * static_cast<float>(X[index]));
  }
  sum = BlockReduceSum<float>(
      sum,
      v_shared,
      min(blockDim.x,
          kCUDABlockReduceNumThreads)); // use BlockReduceSum() to sum X_ij^2

  if (threadIdx.x == 0) {
    rms[i] = static_cast<T>(rsqrt((sum / static_cast<float>(N)) + eps));
  }

  __syncthreads();

  using T_ACC = T;
  for (int64_t j = threadIdx.x; j < N; j += min(blockDim.x, kCUDANumThreads)) {
    const int64_t index = i * N + j;
    Y[index] = static_cast<T_ACC>(X[index]) * static_cast<T_ACC>(rms[i]);
    output[index] = Y[index] * weights[index % N];
  }
}

template <typename T>
void forward_kernel(RMSNormMeta const *m,
                    T const *input_ptr,
                    T const *weight_ptr,
                    T *output_ptr,
                    cudaStream_t stream) {

  std::pair<int, int> kernel1_parallelism =
      std::make_pair(m->batch_size, kCUDABlockReduceNumThreads);
  std::pair<int, int> kernel2_parallelism =
      std::make_pair(m->batch_size, kCUDANumThreads);

  int num_blocks =
      std::max(kernel1_parallelism.first, kernel2_parallelism.first);
  int num_threads =
      std::max(kernel1_parallelism.second, kernel2_parallelism.second);

  RMSNormFusedForwardKernel<T>
      <<<num_blocks, num_threads, 0, stream>>>(m->in_dim,
                                               m->eps,
                                               input_ptr,
                                               static_cast<T *>(m->rms_ptr),
                                               static_cast<T *>(m->norm_ptr),
                                               weight_ptr,
                                               output_ptr);
}

void forward_kernel_wrapper(RMSNormMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorR const &weight,
                            GenericTensorAccessorW const &output) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  assert(output.data_type == input.data_type);
  assert(weight.data_type == output.data_type);
  if (output.data_type == DT_HALF) {
    forward_kernel(m,
                   input.get_half_ptr(),
                   weight.get_half_ptr(),
                   output.get_half_ptr(),
                   stream);
  } else if (output.data_type == DT_FLOAT) {
    forward_kernel(m,
                   input.get_float_ptr(),
                   weight.get_float_ptr(),
                   output.get_float_ptr(),
                   stream);
  } else {
    assert(false && "Unsupported data type");
  }

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[RMSNorm] forward time (CF) = %.2fms\n", elapsed);
  }
}

void inference_kernel_wrapper(RMSNormMeta *m,
                              BatchConfig const *bc,
                              GenericTensorAccessorR const &input,
                              GenericTensorAccessorR const &weight,
                              GenericTensorAccessorW const &output) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  assert(output.data_type == input.data_type);
  assert(weight.data_type == output.data_type);

  // save input activation if needed for PEFT
  if (bc->num_active_peft_tokens() > 0) {
    // check that at most one dimension after the first is > 1. TODO(goliaro):
    // support case where this condition does not hold
    int non_unit_dims_encountered = 0;
    for (int i = 1; i < input.domain.get_dim(); i++) {
      int dim_i = input.domain.hi()[i] - input.domain.lo()[i] + 1;
      if (dim_i > 1) {
        non_unit_dims_encountered++;
      }
    }
    assert(non_unit_dims_encountered <= 1);

    // allocate space for all peft tokens
    MemoryAllocator *allocator = m->handle.peft_activation_allocator;
    int in_dim = input.domain.hi()[0] - input.domain.lo()[0] + 1;
    m->input_activation = allocator->allocate_instance_untyped(
        data_type_size(input.data_type) * bc->num_active_peft_tokens() *
        in_dim);

    int tokens_previous_requests = 0;
    for (int i = 0; i < bc->max_requests_per_batch(); i++) {
      if (bc->request_completed[i]) {
        continue;
      }
      // Skip non-PEFT requests and PEFT forward-only requests
      if (bc->requestsInfo[i].peft_model_id == PEFTModelID::NO_ID ||
          !bc->requestsInfo[i].peft_bwd) {
        tokens_previous_requests += bc->requestsInfo[i].num_tokens_in_batch;
        continue;
      }
      int num_peft_tokens = bc->requestsInfo[i].num_tokens_in_batch;

      if (input.data_type == DT_FLOAT) {
        checkCUDA(cudaMemcpyAsync(
            m->input_activation,
            input.get_float_ptr() + tokens_previous_requests * in_dim,
            data_type_size(input.data_type) * num_peft_tokens * in_dim,
            cudaMemcpyDeviceToDevice,
            stream));
      } else if (input.data_type == DT_HALF) {
        checkCUDA(cudaMemcpyAsync(
            m->input_activation,
            input.get_half_ptr() + tokens_previous_requests * in_dim,
            data_type_size(input.data_type) * num_peft_tokens * in_dim,
            cudaMemcpyDeviceToDevice,
            stream));
      } else {
        assert(false && "unsupport datatype in layernorm");
      }
    }
  }

  if (output.data_type == DT_HALF) {
    forward_kernel(m,
                   input.get_half_ptr(),
                   weight.get_half_ptr(),
                   output.get_half_ptr(),
                   stream);
  } else if (output.data_type == DT_FLOAT) {
    forward_kernel(m,
                   input.get_float_ptr(),
                   weight.get_float_ptr(),
                   output.get_float_ptr(),
                   stream);
  } else {
    assert(false && "Unsupported data type");
  }

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[RMSNorm] forward time (CF) = %.2fms\n", elapsed);
  }
}

template <typename T>
__global__ void ComputeInternalGradientsCUDAKernel(
    int64_t N, T const *dY, T const *X, T const *gamma, T const *rrms, T *c2) {
  __shared__ T ds_storage[C10_WARP_SIZE];
  const int64_t i = blockIdx.x;
  T ds = 0;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    int const index = i * N + j;
    ds += dY[index] * X[index] * gamma[j];
  }
  ds = BlockReduceSum<T>(ds, ds_storage);
  if (threadIdx.x == 0) {
    c2[i] = -ds * (rrms[i] * rrms[i] * rrms[i]) / static_cast<T>((int)N);
  }
}

template <typename T>
__global__ void RMSNormBackwardCUDAKernel(int64_t N,
                                          T const *dY,
                                          T const *X,
                                          T const *gamma,
                                          T const *c1,
                                          T const *c2,
                                          T *dX) {
  const int64_t i = blockIdx.x;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    dX[index] = c1[i] * dY[index] * gamma[j] + c2[i] * X[index];
  }
}

// Assume the batch size will not be very large, direct implementation is the
// most efficient one.
template <typename T>
__global__ void GammaBackwardCUDAKernel(
    int64_t M, int64_t N, T const *dY, T const *X, T const *rrms, T *dg) {
  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N) {
    T sum1 = 0;
    for (int64_t i = 0; i < M; ++i) {
      const int64_t index = i * N + j;
      sum1 += dY[index] * X[index] * rrms[i];
    }
    dg[j] = sum1;
  }
}

template <typename T>
void backward_kernel(RMSNormMeta const *m,
                     T const *output_grad_ptr,
                     T const *input_ptr,
                     T *input_grad_ptr,
                     T const *weight_ptr,
                     T *weight_grad_ptr,
                     cudaStream_t stream) {
  const int64_t M = m->batch_size;
  const int64_t N = m->num_elements;
  ComputeInternalGradientsCUDAKernel<T>
      <<<M, kCUDABlockReduceNumThreads, 0, stream>>>(
          N,
          output_grad_ptr,
          input_ptr,
          weight_ptr,
          static_cast<T *>(m->rms_ptr),
          static_cast<T *>(m->c2_ptr));

  RMSNormBackwardCUDAKernel<T>
      <<<M, kCUDANumThreads, 0, stream>>>(N,
                                          output_grad_ptr,
                                          input_ptr,
                                          weight_ptr,
                                          static_cast<T *>(m->rms_ptr),
                                          static_cast<T *>(m->c2_ptr),
                                          input_grad_ptr);
  const int64_t B = (N + kCUDANumThreads - 1) / kCUDANumThreads;
  GammaBackwardCUDAKernel<T>
      <<<B, kCUDANumThreads, 0, stream>>>(M,
                                          N,
                                          output_grad_ptr,
                                          input_ptr,
                                          static_cast<T *>(m->rms_ptr),
                                          weight_grad_ptr);
}

void backward_kernel_wrapper(RMSNormMeta const *m,
                             GenericTensorAccessorR const &output_grad,
                             GenericTensorAccessorR const &input,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &weight,
                             GenericTensorAccessorW const &weight_grad) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  assert(input_grad.data_type == input.data_type);
  assert(weight_grad.data_type == weight.data_type);
  assert(output_grad.data_type == input.data_type);
  assert(weight.data_type == output_grad.data_type);

  if (output_grad.data_type == DT_HALF) {
    backward_kernel(m,
                    output_grad.get_half_ptr(),
                    input.get_half_ptr(),
                    input_grad.get_half_ptr(),
                    weight.get_half_ptr(),
                    weight_grad.get_half_ptr(),
                    stream);
  } else if (output_grad.data_type == DT_FLOAT) {
    backward_kernel(m,
                    output_grad.get_float_ptr(),
                    input.get_float_ptr(),
                    input_grad.get_float_ptr(),
                    weight.get_float_ptr(),
                    weight_grad.get_float_ptr(),
                    stream);
  } else {
    assert(false && "Unsupported data type");
  }

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[RMSNorm] backward time (CF) = %.2fms\n", elapsed);
  }
}

template <typename T>
void peft_bwd_kernel(RMSNormMeta const *m,
                     BatchConfig const *bc,
                     T const *output_grad_ptr,
                     T *input_grad_ptr,
                     T const *weight_ptr,
                     cudaStream_t stream) {
  for (int i = 0; i < bc->max_requests_per_batch(); i++) {
    if (bc->request_completed[i]) {
      continue;
    }
    // Skip non-PEFT requests
    if (bc->requestsInfo[i].peft_model_id == PEFTModelID::NO_ID) {
      continue;
    }
    // Skip PEFT forward-only requests
    if (!bc->requestsInfo[i].peft_bwd) {
      continue;
    }

    const int64_t M = bc->requestsInfo[i].num_tokens_in_batch;
    const int64_t N = m->num_elements;
    ComputeInternalGradientsCUDAKernel<T>
        <<<M, kCUDABlockReduceNumThreads, 0, stream>>>(
            N,
            output_grad_ptr,
            static_cast<T *>(m->input_activation),
            weight_ptr,
            static_cast<T *>(m->rms_ptr),
            static_cast<T *>(m->c2_ptr));
    RMSNormBackwardCUDAKernel<T><<<M, kCUDANumThreads, 0, stream>>>(
        N,
        output_grad_ptr,
        static_cast<T *>(m->input_activation),
        weight_ptr,
        static_cast<T *>(m->rms_ptr),
        static_cast<T *>(m->c2_ptr),
        input_grad_ptr);
  }
}

void peft_bwd_kernel_wrapper(RMSNormMeta const *m,
                             BatchConfig const *bc,
                             GenericTensorAccessorR const &output_grad,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &weight) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  assert(input_grad.data_type == output_grad.data_type);
  assert(output_grad.data_type == weight.data_type);

  if (output_grad.data_type == DT_HALF) {
    peft_bwd_kernel(m,
                    bc,
                    output_grad.get_half_ptr(),
                    input_grad.get_half_ptr(),
                    weight.get_half_ptr(),
                    stream);
  } else if (output_grad.data_type == DT_FLOAT) {
    peft_bwd_kernel(m,
                    bc,
                    output_grad.get_float_ptr(),
                    input_grad.get_float_ptr(),
                    weight.get_float_ptr(),
                    stream);
  } else {
    assert(false && "Unsupported data type");
  }

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[RMSNorm] peft_bwd time (CF) = %.2fms\n", elapsed);
  }
}

} // namespace RMSNorm
} // namespace Kernels
} // namespace FlexFlow

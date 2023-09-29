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
#include "flexflow/ops/kernels/residual_rms_norm_kernels.h"
#include "flexflow/ops/residual_rms_norm.h"
#include "flexflow/utils/cuda_helper.h"
#include <cublas_v2.h>

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;

#define C10_WARP_SIZE 32
constexpr int kCUDABlockReduceNumThreads = 512;
constexpr int kCUDANumThreads = 256;

ResidualRMSNormMeta::ResidualRMSNormMeta(FFHandler handler,
                                         ResidualRMSNorm const *rms,
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
  size_t norm_ptr_size = num_elements;
  size_t totalSize = (rms_ptr_size + norm_ptr_size) * data_type_size(data_type);
  gpu_mem_allocator.create_legion_instance(reserveInst, totalSize);
  rms_ptr = gpu_mem_allocator.allocate_instance_untyped(
      rms_ptr_size * data_type_size(data_type));
  norm_ptr = gpu_mem_allocator.allocate_instance_untyped(
      norm_ptr_size * data_type_size(data_type));
}
ResidualRMSNormMeta::~ResidualRMSNormMeta(void) {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
}

namespace Kernels {
namespace ResidualRMSNorm {

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
__global__ void ResidualRMSNormFusedForwardKernel(int64_t N,
                                                  float eps,
                                                  T const *X1,
                                                  T const *X2,
                                                  T *X_out,
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
    X_out[index] = X1[index] + X2[index];
    sum +=
        (static_cast<float>(X_out[index]) * static_cast<float>(X_out[index]));
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
    Y[index] = static_cast<T_ACC>(X_out[index]) * static_cast<T_ACC>(rms[i]);
    output[index] = Y[index] * weights[index % N];
  }
}

template <typename T>
void forward_kernel(ResidualRMSNormMeta const *m,
                    T const *input1_ptr,
                    T const *input2_ptr,
                    T const *weight_ptr,
                    T *residual_output_ptr,
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

  ResidualRMSNormFusedForwardKernel<T>
      <<<num_blocks, num_threads, 0, stream>>>(m->in_dim,
                                               m->eps,
                                               input1_ptr,
                                               input2_ptr,
                                               residual_output_ptr,
                                               static_cast<T *>(m->rms_ptr),
                                               static_cast<T *>(m->norm_ptr),
                                               weight_ptr,
                                               output_ptr);
}

void forward_kernel_wrapper(ResidualRMSNormMeta const *m,
                            GenericTensorAccessorR const &input1,
                            GenericTensorAccessorR const &input2,
                            GenericTensorAccessorR const &weight,
                            GenericTensorAccessorW const &residual_output,
                            GenericTensorAccessorW const &output) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  assert(input1.data_type == input2.data_type);
  assert(output.data_type == input1.data_type);
  assert(weight.data_type == output.data_type);
  assert(residual_output.data_type == output.data_type);
  if (output.data_type == DT_HALF) {
    forward_kernel(m,
                   input1.get_half_ptr(),
                   input2.get_half_ptr(),
                   weight.get_half_ptr(),
                   residual_output.get_half_ptr(),
                   output.get_half_ptr(),
                   stream);
  } else if (output.data_type == DT_FLOAT) {
    forward_kernel(m,
                   input1.get_float_ptr(),
                   input2.get_float_ptr(),
                   weight.get_float_ptr(),
                   residual_output.get_float_ptr(),
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
    printf("[ResidualRMSNorm] forward time (CF) = %.2fms\n", elapsed);
  }
}

} // namespace ResidualRMSNorm
} // namespace Kernels
} // namespace FlexFlow

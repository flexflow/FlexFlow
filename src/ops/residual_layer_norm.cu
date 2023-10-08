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
#include "flexflow/ops/residual_layer_norm.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

#define C10_WARP_SIZE 32
constexpr int kCUDABlockReduceNumThreads = 512;
constexpr int kCUDANumThreads = 256;

ResidualLayerNormMeta::ResidualLayerNormMeta(FFHandler handle,
                                             ResidualLayerNorm const *ln,
                                             MemoryAllocator &gpu_mem_allocator)
    : OpMeta(handle) {
  elementwise_affine = ln->elementwise_affine;
  use_bias = ln->use_bias;
  use_two_residuals = ln->use_two_residuals;
  effective_batch_size = ln->effective_batch_size;
  effective_num_elements = ln->effective_num_elements;
  profiling = ln->profiling;
  inference_debugging = ln->inference_debugging;
  eps = ln->eps;
  DataType data_type = ln->data_type;
  size_t totalSize = effective_batch_size * data_type_size(data_type) * 3;
  gpu_mem_allocator.create_legion_instance(reserveInst, totalSize);
  mean_ptr = gpu_mem_allocator.allocate_instance_untyped(
      data_type_size(data_type) * effective_batch_size);
  rstd_ptr = gpu_mem_allocator.allocate_instance_untyped(
      data_type_size(data_type) * effective_batch_size);
  bias_ptr = gpu_mem_allocator.allocate_instance_untyped(
      data_type_size(data_type) * effective_batch_size);
}

ResidualLayerNormMeta::~ResidualLayerNormMeta(void) {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
}

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
  val = (threadIdx.x < min(blockDim.x, max_num_threads) / C10_WARP_SIZE)
            ? shared[lid]
            : 0;
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}

template <typename T>
__global__ void ResidualLayerNormKernel(int64_t N,
                                        float eps,
                                        T const *input_ptr,
                                        T const *residual1_ptr,
                                        T const *residual2_ptr,
                                        T *X,
                                        T *mean,
                                        T *rstd,
                                        T const *gamma,
                                        T const *beta,
                                        T *Y) {
  __shared__ float m_shared[C10_WARP_SIZE];
  __shared__ float v_shared[C10_WARP_SIZE];
  const int64_t i = blockIdx.x;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  for (int64_t j = threadIdx.x; j < N;
       j += min(blockDim.x, kCUDABlockReduceNumThreads)) {
    const int64_t index = i * N + j;
    const T residual2_val = (residual2_ptr == nullptr)
                                ? T(0)
                                : static_cast<T>(residual2_ptr[index]);
    X[index] = input_ptr[index] + residual1_ptr[index] + residual2_val;
    sum1 += static_cast<float>(X[index]);
    sum2 += static_cast<float>(X[index]) * static_cast<float>(X[index]);
  }
  if (threadIdx.x < kCUDABlockReduceNumThreads) {
    sum1 = BlockReduceSum<float>(
        sum1, m_shared, min(blockDim.x, kCUDABlockReduceNumThreads));
    sum2 = BlockReduceSum<float>(
        sum2, v_shared, min(blockDim.x, kCUDABlockReduceNumThreads));
  }
  if (threadIdx.x == 0) {
    float const scale = float(1) / static_cast<float>(N);
    sum1 *= scale;
    sum2 = max(sum2 * scale - sum1 * sum1, float(0));
    mean[i] = static_cast<T>(sum1);
    rstd[i] = static_cast<T>(rsqrt(sum2 + eps));
  }

  __syncthreads();

  using T_ACC = T;
  for (int64_t j = threadIdx.x; j < N; j += min(blockDim.x, kCUDANumThreads)) {
    const int64_t index = i * N + j;
    const T_ACC gamma_v =
        gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[j]);
    const T_ACC beta_v =
        beta == nullptr ? T_ACC(0) : static_cast<T_ACC>(beta[j]);
    Y[index] = (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(mean[i])) *
                   static_cast<T_ACC>(rstd[i]) * gamma_v +
               beta_v;
  }
}

/*static*/
template <typename T>
void ResidualLayerNorm::inference_kernel(ResidualLayerNormMeta const *m,
                                         T const *input_ptr,
                                         T const *residual1_ptr,
                                         T const *residual2_ptr,
                                         T *added_output_ptr,
                                         T *output_ptr,
                                         T const *gamma_ptr,
                                         T const *beta_ptr,
                                         cudaStream_t stream) {

  std::pair<int, int> kernel1_parallelism =
      std::make_pair(m->effective_batch_size, kCUDABlockReduceNumThreads);
  std::pair<int, int> kernel2_parallelism =
      std::make_pair(m->effective_batch_size, kCUDANumThreads);

  int num_blocks =
      std::max(kernel1_parallelism.first, kernel2_parallelism.first);
  int num_threads =
      std::max(kernel1_parallelism.second, kernel2_parallelism.second);

  ResidualLayerNormKernel<T>
      <<<num_blocks, num_threads, 0, stream>>>(m->effective_num_elements,
                                               m->eps,
                                               input_ptr,
                                               residual1_ptr,
                                               residual2_ptr,
                                               added_output_ptr,
                                               static_cast<T *>(m->mean_ptr),
                                               static_cast<T *>(m->rstd_ptr),
                                               gamma_ptr,
                                               beta_ptr,
                                               output_ptr);
}

/*static*/
void ResidualLayerNorm::inference_kernel_wrapper(
    ResidualLayerNormMeta const *m,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorR const &residual1,
    GenericTensorAccessorR const &residual2,
    GenericTensorAccessorW &added_output,
    GenericTensorAccessorW &output,
    GenericTensorAccessorR const &gamma,
    GenericTensorAccessorR const &beta) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  if (m->input_type[0] == DT_FLOAT) {
    ResidualLayerNorm::inference_kernel<float>(
        m,
        input.get_float_ptr(),
        residual1.get_float_ptr(),
        m->use_two_residuals ? residual2.get_float_ptr() : nullptr,
        added_output.get_float_ptr(),
        output.get_float_ptr(),
        m->elementwise_affine ? gamma.get_float_ptr() : nullptr,
        (m->elementwise_affine && m->use_bias) ? beta.get_float_ptr() : nullptr,
        stream);
  } else if (m->input_type[0] == DT_HALF) {
    ResidualLayerNorm::inference_kernel<half>(
        m,
        input.get_half_ptr(),
        residual1.get_half_ptr(),
        m->use_two_residuals ? residual2.get_half_ptr() : nullptr,
        added_output.get_half_ptr(),
        output.get_half_ptr(),
        m->elementwise_affine ? gamma.get_half_ptr() : nullptr,
        (m->elementwise_affine && m->use_bias) ? beta.get_half_ptr() : nullptr,
        stream);
  } else {
    assert(false && "unsupport datatype in layernorm");
  }

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[ResidualLayerNorm] forward time (CF) = %.9fms\n", elapsed);
  }
}

}; // namespace FlexFlow

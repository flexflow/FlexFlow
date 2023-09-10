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
#include "flexflow/ops/add_bias_residual_layer_norm.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

#define C10_WARP_SIZE 32
constexpr int kCUDABlockReduceNumThreads = 512;
constexpr int kCUDANumThreads = 256;

AddBiasResidualLayerNormMeta::AddBiasResidualLayerNormMeta(FFHandler handle,
                             AddBiasResidualLayerNorm const *ln,
                             MemoryAllocator &gpu_mem_allocator)
    : OpMeta(handle) {
  elementwise_affine = ln->elementwise_affine;
  use_bias = ln->use_bias;
  effective_batch_size = ln->effective_batch_size;
  effective_num_elements = ln->effective_num_elements;
  profiling = ln->profiling;
  eps = ln->eps;
  DataType data_type = ln->data_type;
  size_t totalSize = effective_batch_size * data_type_size(data_type) * 6;
  gpu_mem_allocator.create_legion_instance(reserveInst, totalSize);
  mean_ptr = gpu_mem_allocator.allocate_instance_untyped(
      data_type_size(data_type) * effective_batch_size);
  rstd_ptr = gpu_mem_allocator.allocate_instance_untyped(
      data_type_size(data_type) * effective_batch_size);
  ds_ptr = gpu_mem_allocator.allocate_instance_untyped(
      data_type_size(data_type) * effective_batch_size);
  db_ptr = gpu_mem_allocator.allocate_instance_untyped(
      data_type_size(data_type) * effective_batch_size);
  scale_ptr = gpu_mem_allocator.allocate_instance_untyped(
      data_type_size(data_type) * effective_batch_size);
  bias_ptr = gpu_mem_allocator.allocate_instance_untyped(
      data_type_size(data_type) * effective_batch_size);
}

AddBiasResidualLayerNormMeta::~AddBiasResidualLayerNormMeta(void) {
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
__inline__ __device__ T BlockReduceSum(T val, T *shared) {
  int const lid = threadIdx.x % C10_WARP_SIZE;
  int const wid = threadIdx.x / C10_WARP_SIZE;
  val = WarpReduceSum(val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (threadIdx.x < blockDim.x / C10_WARP_SIZE) ? shared[lid] : 0;
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}

template <typename T>
__global__ void RowwiseMomentsCUDAKernel(
    int64_t N, float eps, T const *X, T *mean, T *rstd) {
  __shared__ float m_shared[C10_WARP_SIZE];
  __shared__ float v_shared[C10_WARP_SIZE];
  const int64_t i = blockIdx.x;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    sum1 += static_cast<float>(X[index]);
    sum2 += static_cast<float>(X[index]) * static_cast<float>(X[index]);
  }
  sum1 = BlockReduceSum<float>(sum1, m_shared);
  sum2 = BlockReduceSum<float>(sum2, v_shared);
  if (threadIdx.x == 0) {
    float const scale = float(1) / static_cast<float>(N);
    sum1 *= scale;
    sum2 = max(sum2 * scale - sum1 * sum1, float(0));
    mean[i] = static_cast<T>(sum1);
    rstd[i] = static_cast<T>(rsqrt(sum2 + eps));
  }
}

template <typename T>
__global__ void AddBiasResidualLayerNormForwardCUDAKernel(int64_t N,
                                           T const *X,
                                           T const *mean,
                                           T const *rstd,
                                           T const *gamma,
                                           T const *beta,
                                           T *Y) {
  using T_ACC = T;
  const int64_t i = blockIdx.x;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
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
void AddBiasResidualLayerNorm::inference_kernel(AddBiasResidualLayerNormMeta const *m,
                               T const *in_ptr,
                               T *out_ptr,
                               T const *gamma_ptr,
                               T const *beta_ptr,
                               cudaStream_t stream) {
  RowwiseMomentsCUDAKernel<T>
      <<<m->effective_batch_size, kCUDABlockReduceNumThreads, 0, stream>>>(
          m->effective_num_elements,
          m->eps,
          in_ptr,
          static_cast<T *>(m->mean_ptr),
          static_cast<T *>(m->rstd_ptr));
  AddBiasResidualLayerNormForwardCUDAKernel<T>
      <<<m->effective_batch_size, kCUDANumThreads, 0, stream>>>(
          m->effective_num_elements,
          in_ptr,
          static_cast<T *>(m->mean_ptr),
          static_cast<T *>(m->rstd_ptr),
          gamma_ptr,
          beta_ptr,
          out_ptr);
}

/*static*/
void AddBiasResidualLayerNorm::inference_kernel_wrapper(AddBiasResidualLayerNormMeta const *m,
                                       GenericTensorAccessorR const &input,
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
    AddBiasResidualLayerNorm::inference_kernel<float>(m,
                                     input.get_float_ptr(),
                                     output.get_float_ptr(),
                                     gamma.get_float_ptr(),
                                     m->use_bias ? beta.get_float_ptr()
                                                 : nullptr,
                                     stream);
  } else if (m->input_type[0] == DT_HALF) {
    AddBiasResidualLayerNorm::inference_kernel<half>(m,
                                    input.get_half_ptr(),
                                    output.get_half_ptr(),
                                    gamma.get_half_ptr(),
                                    m->use_bias ? beta.get_half_ptr() : nullptr,
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
    printf("[AddBiasResidualLayerNorm] forward time (CF) = %.2fms\n", elapsed);
    // print_tensor<T>(in_ptr, 32, "[AddBiasResidualLayerNorm:forward:input]");
    // print_tensor<T>(out_ptr, 32, "[AddBiasResidualLayerNorm:forward:output]");
  }
}

template <typename T>
__global__ void ComputeInternalGradientsCUDAKernel(
    int64_t N, T const *dY, T const *X, T const *gamma, T *ds, T *db) {
  using T_ACC = T;
  __shared__ T_ACC ds_shared[C10_WARP_SIZE];
  __shared__ T_ACC db_shared[C10_WARP_SIZE];
  const int64_t i = blockIdx.x;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    const T_ACC gamma_v =
        gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[j]);
    sum1 +=
        static_cast<T_ACC>(dY[index]) * static_cast<T_ACC>(X[index]) * gamma_v;
    sum2 += static_cast<T_ACC>(dY[index]) * gamma_v;
  }
  sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);
  sum2 = BlockReduceSum<T_ACC>(sum2, db_shared);
  if (threadIdx.x == 0) {
    ds[i] = sum1;
    db[i] = sum2;
  }
}

template <typename T>
__global__ void ComputeGradientFusedParamsCUDAKernel(int64_t M,
                                                     int64_t N,
                                                     T const *mean,
                                                     T const *rstd,
                                                     T const *ds,
                                                     T const *db,
                                                     T *c1,
                                                     T *c2) {
  using T_ACC = T;
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < M) {
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(N);
    const T_ACC a = (db[index] * static_cast<T_ACC>(mean[index]) - ds[index]) *
                    static_cast<T_ACC>(rstd[index]) *
                    static_cast<T_ACC>(rstd[index]) *
                    static_cast<T_ACC>(rstd[index]) * s;
    c1[index] = a;
    c2[index] = -(a * static_cast<T_ACC>(mean[index]) +
                  db[index] * static_cast<T_ACC>(rstd[index]) * s);
  }
}

}; // namespace FlexFlow

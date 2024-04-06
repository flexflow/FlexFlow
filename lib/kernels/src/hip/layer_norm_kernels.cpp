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

#include "kernels/layer_norm_kernels.h"
#include "kernels/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

#define C10_WARP_SIZE 32
constexpr int kCUDABlockReduceNumThreads = 512;
constexpr int kCUDANumThreads = 256;
constexpr int kColwiseReduceTileSize = 32;

LayerNormPerDeviceState::LayerNormPerDeviceState(
    FFHandler handle,
    bool elementwise_affine_,
    int64_t effective_batch_size_,
    int64_t effective_num_elements_,
    bool profiling_,
    float eps_)
    : PerDeviceOpState(handle) {
  elementwise_affine = elementwise_affine_;
  effective_batch_size = effective_batch_size_;
  effective_num_elements = effective_num_elements_;
  profiling = profiling_;
  eps = eps_;
  checkCUDA(hipMalloc(&mean_ptr, sizeof(float) * effective_batch_size));
  checkCUDA(hipMalloc(&rstd_ptr, sizeof(float) * effective_batch_size));
  checkCUDA(hipMalloc(&ds_ptr, sizeof(float) * effective_batch_size));
  checkCUDA(hipMalloc(&db_ptr, sizeof(float) * effective_batch_size));
  checkCUDA(hipMalloc(&scale_ptr, sizeof(float) * effective_batch_size));
  checkCUDA(hipMalloc(&bias_ptr, sizeof(float) * effective_batch_size));
}

namespace Kernels {
namespace LayerNorm {

LayerNormPerDeviceState init_kernel(PerDeviceFFHandle const &handle,
                                    Allocator const &allocator,
                                    bool elementwise_affine_,
                                    int64_t effective_batch_size_,
                                    int64_t effective_num_elements_,
                                    float eps_) {
  elementwise_affine = elementwise_affine_;
  effective_batch_size = effective_batch_size_;
  effective_num_elements = effective_num_elements_;
  eps = eps_;
  mean = allocator.allocate(sizeof(float) * effective_batch_size);
  rstd = allocator.allocate(sizeof(float) * effective_batch_size);
  ds = allocator.allocate(sizeof(float) * effective_batch_size);
  db = allocator.allocate(sizeof(float) * effective_batch_size);
  scale = allocator.allocate(sizeof(float) * effective_batch_size);
  bias = allocator.allocate(sizeof(float) * effective_batch_size);
  LayerNormPerDeviceState per_device_state =
      LayerNormPerDeviceState(handle,
                              elementwise_affine,
                              effective_batch_size,
                              effective_num_elements,
                              eps,
                              mean,
                              rstd,
                              ds,
                              db,
                              scale,
                              bias);
  return per_device_state;
}

template <DataType T>
struct ForwardKernel {
  void operator()(hipStream_t stream,
                  LayerNormPerDeviceState const *m,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output,
                  GenericTensorAccessorW const &gamma,
                  GenericTensorAccessorW const &beta) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(RowwiseMomentsCUDAKernel<float>),
                       m->effective_batch_size,
                       kCUDABlockReduceNumThreads,
                       0,
                       stream,
                       m->effective_num_elements,
                       m->eps,
                       input.get<T>(),
                       m->mean_ptr,
                       m->rstd_ptr);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(LayerNormForwardCUDAKernel<float>),
                       m->effective_batch_size,
                       kCUDANumThreads,
                       0,
                       stream,
                       m->effective_num_elements,
                       input.get<T>(),
                       m->mean_ptr,
                       m->rstd_ptr,
                       gamma.get<T>(),
                       beta.get<T>(),
                       output.get<T>());
  }
}

template <DataType T>
struct BackwardKernel {
  void operator()(hipStream_t stream,
                  LayerNormPerDeviceState const *m,
                  GenericTensorAccessorR const &output_grad,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &input_grad,
                  GenericTensorAccessorR const &gamma,
                  GenericTensorAccessorW const &gamma_grad,
                  GenericTensorAccessorW const &beta_grad) {
    const int64_t M = m->effective_batch_size;
    const int64_t N = m->effective_num_elements;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(ComputeInternalGradientsCUDAKernel<T>),
                       M,
                       kCUDABlockReduceNumThreads,
                       0,
                       stream,
                       N,
                       output_grad.get<T>(),
                       input.get<T>(),
                       gamma.get<T>(),
                       m->ds_ptr,
                       m->db_ptr);
    const int64_t B = (M + kCUDANumThreads - 1) / kCUDANumThreads;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(ComputeGradientFusedParamsCUDAKernel<T>),
                       B,
                       kCUDANumThreads,
                       0,
                       stream,
                       M,
                       N,
                       m->mean_ptr,
                       m->rstd_ptr,
                       m->ds_ptr,
                       m->db_ptr,
                       m->scale_ptr,
                       m->bias_ptr);
    if (gamma_grad.get<T>() != NULL || beta_grad.get<T>() != NULL) {
      if (M < 512) {
        // For small batch size, do colwise reduce directly
        const int64_t B = (N + kCUDANumThreads - 1) / kCUDANumThreads;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(GammaBetaBackwardSimpleCUDAKernel<T>),
            B,
            kCUDANumThreads,
            0,
            stream,
            M,
            N,
            output_grad.get<T>(),
            input.get<T>(),
            m->mean_ptr,
            m->rstd_ptr,
            gamma_grad.get<T>(),
            beta_grad.get<T>());
      } else {
        const int64_t B =
            (N + kColwiseReduceTileSize - 1) / kColwiseReduceTileSize;
        constexpr int kThreadX = kColwiseReduceTileSize;
        constexpr int kThreadY = kColwiseReduceTileSize / 2;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(GammaBetaBackwardCUDAKernel<T>),
                           B,
                           dim3(kThreadX, kThreadY),
                           0,
                           stream,
                           M,
                           N,
                           output_grad.get<T>(),
                           input.get<T>(),
                           m->mean_ptr,
                           m->rstd_ptr,
                           gamma_grad.get<T>(),
                           beta_grad.get<T>());
      }
    }
  }

  void forward_kernel(hipStream_t stream,
                      LayerNormPerDeviceState const *m,
                      GenericTensorAccessorR const &input,
                      GenericTensorAccessorW const &output,
                      GenericTensorAccessorW const &gamma,
                      GenericTensorAccessorW const &beta) {
    DataTypeDispatch1<ForwardKernel>{}(
        m->data_type, stream, m, input, output, gamma, beta);
  }

  void backward_kernel(hipStream_t stream,
                       LayerNormPerDeviceState const *m,
                       GenericTensorAccessorR const &output_grad,
                       GenericTensorAccessorR const &input,
                       GenericTensorAccessorW const &input_grad,
                       GenericTensorAccessorR const &gamma,
                       GenericTensorAccessorW const &gamma_grad,
                       GenericTensorAccessorW const &beta_grad) {
    DataTypeDispatch1<BackwardKernel>{}(m->data_type,
                                        stream,
                                        m,
                                        output_grad,
                                        input,
                                        input_grad,
                                        gamma,
                                        gamma_grad,
                                        beta_grad);
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
__global__ void
    RowwiseMomentsCUDAKernel(int64_t N, T eps, T const *X, T *mean, T *rstd) {
  __shared__ T m_shared[C10_WARP_SIZE];
  __shared__ T v_shared[C10_WARP_SIZE];
  const int64_t i = blockIdx.x;
  T sum1 = 0;
  T sum2 = 0;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    sum1 += static_cast<T>(X[index]);
    sum2 += static_cast<T>(X[index]) * static_cast<T>(X[index]);
  }
  sum1 = BlockReduceSum<T>(sum1, m_shared);
  sum2 = BlockReduceSum<T>(sum2, v_shared);
  if (threadIdx.x == 0) {
    const T scale = T(1) / static_cast<T>(N);
    sum1 *= scale;
    sum2 = max(sum2 * scale - sum1 * sum1, T(0));
    mean[i] = sum1;
    rstd[i] = rsqrt(sum2 + static_cast<T>(eps));
  }
}

template <typename T>
__global__ void LayerNormForwardCUDAKernel(int64_t N,
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
    sum1 += static_cast<T_ACC>(dY[index]) * static_cast<T_ACC>(X[index]) *
            gamma_v;
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
    const T_ACC a =
        (db[index] * static_cast<T_ACC>(mean[index]) - ds[index]) *
        static_cast<T_ACC>(rstd[index]) * static_cast<T_ACC>(rstd[index]) *
        static_cast<T_ACC>(rstd[index]) * s;
    c1[index] = a;
    c2[index] = -(a * static_cast<T_ACC>(mean[index]) +
                  db[index] * static_cast<T_ACC>(rstd[index]) * s);
  }
}

template <typename T>
__global__ void LayerNormBackwardCUDAKernel(int64_t N,
                                            T const *dY,
                                            T const *X,
                                            T const *gamma,
                                            T const *a,
                                            T const *b,
                                            T const *c,
                                            T *dX) {
  using T_ACC = T;
  const int64_t i = blockIdx.x;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    const T_ACC gamma_v =
        gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[j]);
    dX[index] =
        static_cast<T_ACC>(a[i]) * static_cast<T_ACC>(dY[index]) * gamma_v +
        b[i] * static_cast<T_ACC>(X[index]) + c[i];
  }
}

template <typename T>
__global__ void GammaBetaBackwardSimpleCUDAKernel(int64_t M,
                                                  int64_t N,
                                                  T const *dY,
                                                  T const *X,
                                                  T const *mean,
                                                  T const *rstd,
                                                  T *dg,
                                                  T *db) {
  using T_ACC = T;
  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N) {
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (int64_t i = 0; i < M; ++i) {
      const int64_t index = i * N + j;
      sum1 += dg == nullptr ? T_ACC(0)
                            : static_cast<T_ACC>(dY[index]) *
                                  (static_cast<T_ACC>(X[index]) -
                                    static_cast<T_ACC>(mean[i])) *
                                  static_cast<T_ACC>(rstd[i]);
      sum2 += db == nullptr ? T_ACC(0) : static_cast<T_ACC>(dY[index]);
    }
    if (dg != nullptr) {
      dg[j] = sum1;
    }
    if (db != nullptr) {
      db[j] = sum2;
    }
  }
}

template <typename T>
__global__ void GammaBetaBackwardCUDAKernel(int64_t M,
                                            int64_t N,
                                            T const *dY,
                                            T const *X,
                                            T const *mean,
                                            T const *rstd,
                                            T *dg,
                                            T *db) {
  using T_ACC = T;
  __shared__ T_ACC
      g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize + 1];
  __shared__ T_ACC
      b_shared[kColwiseReduceTileSize][kColwiseReduceTileSize + 1];
  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  T_ACC dg_sum1 = 0;
  T_ACC dg_sum2 = 0;
  T_ACC db_sum1 = 0;
  T_ACC db_sum2 = 0;
  if (j < N) {
    for (int64_t i = threadIdx.y; i < M; i += blockDim.y * 2) {
      const int64_t i1 = i;
      const int64_t i2 = i + blockDim.y;
      const int64_t index1 = i1 * N + j;
      const int64_t index2 = i2 * N + j;
      dg_sum1 += dg == nullptr ? T_ACC(0)
                                : static_cast<T_ACC>(dY[index1]) *
                                      (static_cast<T_ACC>(X[index1]) -
                                      static_cast<T_ACC>(mean[i1])) *
                                      static_cast<T_ACC>(rstd[i1]);
      db_sum1 += db == nullptr ? T_ACC(0) : static_cast<T_ACC>(dY[index1]);
      if (i2 < M) {
        dg_sum2 += dg == nullptr ? T_ACC(0)
                                  : static_cast<T_ACC>(dY[index2]) *
                                        (static_cast<T_ACC>(X[index2]) -
                                        static_cast<T_ACC>(mean[i2])) *
                                        static_cast<T_ACC>(rstd[i2]);
        db_sum2 += db == nullptr ? T_ACC(0) : static_cast<T_ACC>(dY[index2]);
      }
    }
  }
  g_shared[threadIdx.y][threadIdx.x] = dg_sum1;
  g_shared[threadIdx.y + blockDim.y][threadIdx.x] = dg_sum2;
  b_shared[threadIdx.y][threadIdx.x] = db_sum1;
  b_shared[threadIdx.y + blockDim.y][threadIdx.x] = db_sum2;
  __syncthreads();
  T_ACC sum1 = g_shared[threadIdx.x][threadIdx.y];
  T_ACC sum2 = b_shared[threadIdx.x][threadIdx.y];
  sum1 = WarpReduceSum(sum1);
  sum2 = WarpReduceSum(sum2);
  if (threadIdx.x == 0) {
    const int64_t j = blockIdx.x * blockDim.x + threadIdx.y;
    if (j < N) {
      if (dg != nullptr) {
        dg[j] = sum1;
      }
      if (db != nullptr) {
        db[j] = sum2;
      }
    }
  }
  sum1 = g_shared[threadIdx.x][threadIdx.y + blockDim.y];
  sum2 = b_shared[threadIdx.x][threadIdx.y + blockDim.y];
  sum1 = WarpReduceSum(sum1);
  sum2 = WarpReduceSum(sum2);
  if (threadIdx.x == 0) {
    const int64_t j = blockIdx.x * blockDim.x + threadIdx.y + blockDim.y;
    if (j < N) {
      if (dg != nullptr) {
        dg[j] = sum1;
      }
      if (db != nullptr) {
        db[j] = sum2;
      }
    }
  }
}

} // namespace LayerNorm
} // namespace Kernels
} // namespace FlexFlow

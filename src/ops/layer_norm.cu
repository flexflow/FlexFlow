/* Copyright 2021 CMU, Facebook
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

#include "flexflow/ops/layer_norm.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::Rect;
using Legion::PhysicalRegion;
using Legion::coord_t;
using Legion::Memory;
using Legion::Machine;

#define C10_WARP_SIZE 32
constexpr int kCUDABlockReduceNumThreads = 512;
constexpr int kCUDANumThreads = 256;
constexpr int kColwiseReduceTileSize = 32;

#ifdef DEADCODE
void LayerNorm::create_weights(FFModel& model)
{
  std::string pcname = name;
  task_is = model.get_or_create_task_is(outputs[0].numDim, pcname);

  // TODO: temp work, will let users to pick either NCCL or PS
#ifdef FF_USE_NCCL
  ParameterSyncType comm_type = ParameterSyncType::NCCL;
#else
  ParameterSyncType comm_type = ParameterSyncType::PS;
#endif

  // Create scale and bias
  Initializer* scale_initializer = new ConstantInitializer(1.0f);
  Initializer* bias_initializer = new ConstantInitializer(0.0f);
  const int dims[1] = {weights[0].adim[0]};
  switch (outputs[0].numDim) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      weights[0] = model.create_linear_weight<1, DIM>(this, dims, DT_FLOAT, \
          scale_initializer, true/*create_grad*/, comm_type); \
      weights[1] = model.create_linear_weight<1, DIM>(this, dims, DT_FLOAT, \
          bias_initializer, true/*create_grad*/, comm_type); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
  }
}

void LayerNorm::create_output_and_partition(FFModel& model)
{
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = model.get_or_create_task_is(outputs[0].numDim, pcname);
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Domain part_rect = runtime->get_index_space_domain(ctx, task_is);
  {
    int dims[MAX_TENSOR_DIM];
    int ndims = outputs[0].numDim;
    for (int i = 0; i < outputs[0].numDim; i++)
      dims[i] = outputs[0].adim[ndims-1-i];
    switch (ndims) {
#define DIMFUNC(DIM) \
      case DIM: \
      { \
        outputs[0] = model.create_tensor<DIM>(dims, outputs[0].data_type, this); \
        break; \
      }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    }
    outputs[0].owner_op = this;
    outputs[0].owner_idx = 0;
  }
  Domain input_rect = runtime->get_index_partition_color_space(
      ctx, inputs[0].part.get_index_partition());
  // Currently assume output and input must be partitioned in the same way
  if (input_rect == part_rect) {
    input_lps[0] = inputs[0].part;
    input_grad_lps[0] = inputs[0].part_grad;
  } else {
    assert(false && "LayerNorm currently assume output/input have same partition");
  }
}
#endif

LayerNormMeta::LayerNormMeta(FFHandler handle, const LayerNorm* ln)
: OpMeta(handle)
{
  elementwise_affine = ln->elementwise_affine;
  effective_batch_size = ln->effective_batch_size;
  effective_num_elements = ln->effective_num_elements;
  eps = ln->eps;
  checkCUDA(cudaMalloc(&mean_ptr, sizeof(float) * effective_batch_size));
  checkCUDA(cudaMalloc(&rstd_ptr, sizeof(float) * effective_batch_size));
  checkCUDA(cudaMalloc(&ds_ptr, sizeof(float) * effective_batch_size));
  checkCUDA(cudaMalloc(&db_ptr, sizeof(float) * effective_batch_size));
  checkCUDA(cudaMalloc(&scale_ptr, sizeof(float) * effective_batch_size));
  checkCUDA(cudaMalloc(&bias_ptr, sizeof(float) * effective_batch_size));
}

__host__
OpMeta* LayerNorm::init_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)
{
  LayerNorm* ln = (LayerNorm*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  LayerNormMeta* meta = new LayerNormMeta(handle, ln);
  return meta;
}

/*
  regions[0](I): input
  regions[1](O): output
  regions[2](I/O): gamma
  regions[3](I/O): beta
*/
void LayerNorm::forward_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)
{
  const LayerNormMeta* m = *((LayerNormMeta**) task->local_args);
  assert(task->regions.size() == regions.size());
  const float *in_ptr = NULL;
  float *out_ptr = NULL, *gamma_ptr = NULL, *beta_ptr = NULL;
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  in_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  out_ptr = helperGetTensorPointerWO<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  assert(in_domain == out_domain);
  assert(in_domain.get_volume() == m->effective_num_elements * m->effective_batch_size);
  if (m->elementwise_affine) {
    assert(regions.size() == 4);
    Domain gamma_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
    gamma_ptr = helperGetTensorPointerRW<float>(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
    Domain beta_domain = runtime->get_index_space_domain(
      ctx, task->regions[3].region.get_index_space());
    beta_ptr = helperGetTensorPointerRW<float>(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);
    assert(gamma_domain == beta_domain);
    assert(gamma_domain.get_volume() == m->effective_num_elements);
  } else {
    assert(regions.size() == 2);
  }

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  forward_kernel<float>(m, in_ptr, out_ptr, gamma_ptr, beta_ptr, stream);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff)
{
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
__inline__ __device__ T BlockReduceSum(T val, T* shared) {
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
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
    int64_t N,
    T eps,
    const T* X,
    T* mean,
    T* rstd) {
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
__global__ void LayerNormForwardCUDAKernel(
    int64_t N,
    const T* X,
    const T* mean,
    const T* rstd,
    const T* gamma,
    const T* beta,
    T* Y) {
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

template<typename T>
void LayerNorm::forward_kernel(const LayerNormMeta* m,
                               const T* in_ptr,
                               T* out_ptr,
                               T* gamma_ptr,
                               T* beta_ptr,
                               cudaStream_t stream)
{
  RowwiseMomentsCUDAKernel<float>
      <<<m->effective_batch_size, kCUDABlockReduceNumThreads, 0, stream>>>(
          m->effective_num_elements, m->eps, in_ptr, m->mean_ptr, m->rstd_ptr);
  LayerNormForwardCUDAKernel<float><<<m->effective_batch_size, kCUDANumThreads, 0, stream>>>(
      m->effective_num_elements, in_ptr, m->mean_ptr, m->rstd_ptr, gamma_ptr, beta_ptr, out_ptr);
}

template <typename T>
__global__ void ComputeInternalGradientsCUDAKernel(
    int64_t N,
    const T* dY,
    const T* X,
    const T* gamma,
    T* ds,
    T* db) {
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
__global__ void ComputeGradientFusedParamsCUDAKernel(
    int64_t M,
    int64_t N,
    const T* mean,
    const T* rstd,
    const T* ds,
    const T* db,
    T* c1,
    T* c2) {
  using T_ACC = T;
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < M) {
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(N);
    const T_ACC a = (db[index] * static_cast<T_ACC>(mean[index]) - ds[index]) *
        static_cast<T_ACC>(rstd[index]) * static_cast<T_ACC>(rstd[index]) *
        static_cast<T_ACC>(rstd[index]) * s;
    c1[index] = a;
    c2[index] =
        -(a * static_cast<T_ACC>(mean[index]) +
          db[index] * static_cast<T_ACC>(rstd[index]) * s);
  }
}

template <typename T>
__global__ void LayerNormBackwardCUDAKenrel(
    int64_t N,
    const T* dY,
    const T* X,
    const T* gamma,
    const T* a,
    const T* b,
    const T* c,
    T* dX) {
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
__global__ void GammaBetaBackwardSimpleCUDAKernel(
    int64_t M,
    int64_t N,
    const T* dY,
    const T* X,
    const T* mean,
    const T* rstd,
    T* dg,
    T* db) {
  using T_ACC = T;
  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N) {
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (int64_t i = 0; i < M; ++i) {
      const int64_t index = i * N + j;
      sum1 += dg == nullptr ? T_ACC(0)
                            : static_cast<T_ACC>(dY[index]) *
              (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(mean[i])) *
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
__global__ void GammaBetaBackwardCUDAKernel(
    int64_t M,
    int64_t N,
    const T* dY,
    const T* X,
    const T* mean,
    const T* rstd,
    T* dg,
    T* db) {
  using T_ACC = T;
  __shared__ T_ACC g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize + 1];
  __shared__ T_ACC b_shared[kColwiseReduceTileSize][kColwiseReduceTileSize + 1];
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
              (static_cast<T_ACC>(X[index1]) - static_cast<T_ACC>(mean[i1])) *
              static_cast<T_ACC>(rstd[i1]);
      db_sum1 += db == nullptr ? T_ACC(0) : static_cast<T_ACC>(dY[index1]);
      if (i2 < M) {
        dg_sum2 += dg == nullptr ? T_ACC(0)
                                 : static_cast<T_ACC>(dY[index2]) *
                (static_cast<T_ACC>(X[index2]) - static_cast<T_ACC>(mean[i2])) *
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

/*
  regions[0](I): output_grad
  regions[1](I): input
  regions[2](I/O): input_grad
  regions[3](I): gamma
  regions[4](I/O): gamma_grad
  regions[5](I/O): beta_grad
   */
void LayerNorm::backward_task(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime *runtime) {
  const LayerNormMeta* m = *((LayerNormMeta**) task->local_args);
  assert(task->regions.size() == regions.size());
  const float *in_ptr = NULL, *out_grad_ptr = NULL, *gamma_ptr = NULL;
  float *in_grad_ptr = NULL, *gamma_grad_ptr = NULL, *beta_grad_ptr = NULL;
  Domain out_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  out_grad_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  in_ptr = helperGetTensorPointerRO<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  Domain in_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  in_grad_ptr = helperGetTensorPointerRW<float>(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  assert(in_domain == out_grad_domain);
  assert(in_domain.get_volume() == m->effective_num_elements * m->effective_batch_size);
  if (m->elementwise_affine) {
    assert(regions.size() == 6);
    Domain gamma_domain = runtime->get_index_space_domain(
      ctx, task->regions[3].region.get_index_space());
    gamma_ptr = helperGetTensorPointerRO<float>(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);
    Domain gamma_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[4].region.get_index_space());
    gamma_grad_ptr = helperGetTensorPointerRW<float>(
      regions[4], task->regions[4], FID_DATA, ctx, runtime);
    Domain beta_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[5].region.get_index_space());
    beta_grad_ptr = helperGetTensorPointerRW<float>(
      regions[5], task->regions[5], FID_DATA, ctx, runtime);
    assert(gamma_domain == gamma_grad_domain);
    assert(gamma_domain == beta_grad_domain);
    assert(gamma_domain.get_volume() == m->effective_num_elements);
  } else {
    assert(regions.size() == 3);
  }

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  backward_kernel<float>(m, out_grad_ptr, in_ptr, in_grad_ptr,
      gamma_ptr, gamma_grad_ptr, beta_grad_ptr, stream);
}

template<typename T>
void LayerNorm::backward_kernel(const LayerNormMeta* m,
                                const T* output_grad_ptr,
                                const T* input_ptr,
                                T* input_grad_ptr,
                                const T* gamma_ptr,
                                T* gamma_grad_ptr,
                                T* beta_grad_ptr,
                                cudaStream_t stream)
{
  const int64_t M = m->effective_batch_size;
  const int64_t N = m->effective_num_elements;
  ComputeInternalGradientsCUDAKernel<T>
      <<<M, kCUDABlockReduceNumThreads, 0, stream>>>(
          N, output_grad_ptr, input_ptr, gamma_ptr, m->ds_ptr, m->db_ptr);
  const int64_t B = (M + kCUDANumThreads - 1) / kCUDANumThreads;
  ComputeGradientFusedParamsCUDAKernel<T>
      <<<B, kCUDANumThreads, 0, stream>>>(
          M,
          N,
          m->mean_ptr,
          m->rstd_ptr,
          m->ds_ptr,
          m->db_ptr,
          m->scale_ptr,
          m->bias_ptr);
  if (gamma_grad_ptr != NULL || beta_grad_ptr != NULL) {
    if (M < 512) {
      // For small batch size, do colwise reduce directly
      const int64_t B = (N + kCUDANumThreads - 1) / kCUDANumThreads;
      GammaBetaBackwardSimpleCUDAKernel<T>
          <<<B, kCUDANumThreads, 0, stream>>>(
              M,
              N,
              output_grad_ptr,
              input_ptr,
              m->mean_ptr,
              m->rstd_ptr,
              gamma_grad_ptr,
              beta_grad_ptr);
    } else {
      const int64_t B =
          (N + kColwiseReduceTileSize - 1) / kColwiseReduceTileSize;
      constexpr int kThreadX = kColwiseReduceTileSize;
      constexpr int kThreadY = kColwiseReduceTileSize / 2;
      GammaBetaBackwardCUDAKernel<T>
          <<<B, dim3(kThreadX, kThreadY), 0, stream>>>(
              M,
              N,
              output_grad_ptr,
              input_ptr,
              m->mean_ptr,
              m->rstd_ptr,
              gamma_grad_ptr,
              beta_grad_ptr);

    }
  }
}

bool LayerNorm::measure_operator_cost(
    Simulator* sim,
    const ParallelConfig& pc,
    CostMetrics& cost_metrics) const
{
  return false;
}

}; //namespace FlexFlow

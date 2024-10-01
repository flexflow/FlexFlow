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
constexpr int kColwiseReduceTileSize = 32;

AddBiasResidualLayerNormMeta::AddBiasResidualLayerNormMeta(
    FFHandler handle,
    AddBiasResidualLayerNorm const *ln,
    MemoryAllocator &gpu_mem_allocator)
    : OpMeta(handle, ln) {
  elementwise_affine = ln->elementwise_affine;
  use_bias = ln->use_bias;
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
  allocated_peft_buffer_size = 0;
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
  val = (threadIdx.x < (blockDim.x / C10_WARP_SIZE)) ? shared[lid] : T(0);
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}

template <typename T>
__global__ void LayerNormFusedForwardKernel(int64_t N,
                                            int64_t attn_bias_dim,
                                            float eps,
                                            T const *input_ptr,
                                            T const *attn_bias_ptr,
                                            T const *residual_ptr,
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
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    const int64_t bias_idx = index % attn_bias_dim;
    X[index] = input_ptr[index] + attn_bias_ptr[bias_idx] + residual_ptr[index];
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

  __syncthreads();

  using T_ACC = T;
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
void AddBiasResidualLayerNorm::inference_kernel(
    AddBiasResidualLayerNormMeta const *m,
    int attn_bias_dim,
    int residual_volume,
    T const *input_ptr,
    T const *attn_bias_ptr,
    T const *residual_ptr,
    T *added_output_ptr,
    T *output_ptr,
    T const *gamma_ptr,
    T const *beta_ptr,
    cudaStream_t stream) {
  LayerNormFusedForwardKernel<T>
      <<<m->effective_batch_size,
         std::min(CUDA_NUM_THREADS, (int)m->effective_num_elements),
         0,
         stream>>>(m->effective_num_elements,
                   attn_bias_dim,
                   m->eps,
                   input_ptr,
                   attn_bias_ptr,
                   residual_ptr,
                   added_output_ptr,
                   static_cast<T *>(m->mean_ptr),
                   static_cast<T *>(m->rstd_ptr),
                   gamma_ptr,
                   beta_ptr,
                   output_ptr);
}

/*static*/
void AddBiasResidualLayerNorm::inference_kernel_wrapper(
    AddBiasResidualLayerNormMeta *m,
    BatchConfig const *bc,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorR const &attn_bias,
    GenericTensorAccessorR const &residual,
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
  // save input activation if needed for PEFT
  if (bc->num_active_peft_tokens() > 0) {
    // Check that we have at most one request that requires peft_bwd
    int num_peft_requests = 0;
    for (int i = 0; i < bc->max_requests_per_batch(); i++) {
      if (bc->request_completed[i]) {
        continue;
      }
      if (bc->requestsInfo[i].peft_model_id == PEFTModelID::NO_ID) {
        continue;
      }
      if (bc->requestsInfo[i].peft_bwd) {
        num_peft_requests++;
      }
    }
    assert(num_peft_requests <= 1);

    for (int i = 0; i < bc->max_requests_per_batch(); i++) {
      if (bc->request_completed[i]) {
        continue;
      }
      // Skip non-PEFT requests
      if (bc->requestsInfo[i].peft_model_id == PEFTModelID::NO_ID) {
        continue;
      }
      int num_peft_tokens = bc->requestsInfo[i].num_tokens_in_batch;
      int max_peft_tokens = bc->requestsInfo[i].max_length;
      int first_token_offset = bc->requestsInfo[i].first_token_offset_in_batch;
      int in_dim = input.domain.hi()[0] - input.domain.lo()[0] + 1;
      if (bc->requestsInfo[i].peft_bwd) {
        size_t activation_size_needed =
            data_type_size(m->input_type[0]) * max_peft_tokens * in_dim;
        if (activation_size_needed > m->allocated_peft_buffer_size) {
          MemoryAllocator *allocator = m->handle.peft_activation_allocator;
          m->input_activation =
              allocator->allocate_instance_untyped(activation_size_needed);
          m->allocated_peft_buffer_size = activation_size_needed;
        }
        // copy input activation
        if (m->input_type[0] == DT_FLOAT) {
          checkCUDA(cudaMemcpyAsync(
              m->input_activation,
              added_output.get_float_ptr() + first_token_offset * in_dim,
              data_type_size(m->input_type[0]) * num_peft_tokens * in_dim,
              cudaMemcpyDeviceToDevice,
              stream));
        } else if (m->input_type[0] == DT_HALF) {
          checkCUDA(cudaMemcpyAsync(
              m->input_activation,
              added_output.get_half_ptr() + first_token_offset * in_dim,
              data_type_size(m->input_type[0]) * num_peft_tokens * in_dim,
              cudaMemcpyDeviceToDevice,
              stream));
        } else {
          assert(false && "unsupport datatype in layernorm");
        }
      }
    }
  }

  // inference kernel
  int attn_bias_dim = attn_bias.domain.hi()[0] - attn_bias.domain.lo()[0] + 1;
  int residual_volume = residual.domain.get_volume();
  if (m->input_type[0] == DT_FLOAT) {
    AddBiasResidualLayerNorm::inference_kernel<float>(
        m,
        attn_bias_dim,
        residual_volume,
        input.get_float_ptr(),
        attn_bias.get_float_ptr(),
        residual.get_float_ptr(),
        added_output.get_float_ptr(),
        output.get_float_ptr(),
        m->elementwise_affine ? gamma.get_float_ptr() : nullptr,
        (m->elementwise_affine && m->use_bias) ? beta.get_float_ptr() : nullptr,
        stream);
  } else if (m->input_type[0] == DT_HALF) {
    AddBiasResidualLayerNorm::inference_kernel<half>(
        m,
        attn_bias_dim,
        residual_volume,
        input.get_half_ptr(),
        attn_bias.get_half_ptr(),
        residual.get_half_ptr(),
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
    printf("[AddBiasResidualLayerNorm] forward time (CF) = %.9fms\n", elapsed);
    // if (m->input_type[0] == DT_FLOAT) {
    //   print_tensor<float>(input.get_float_ptr(),
    //                       32,
    //                       "[AddBiasResidualLayerNorm:forward:input]");
    //   print_tensor<float>(attn_bias.get_float_ptr(),
    //                       32,
    //                       "[AddBiasResidualLayerNorm:forward:attn_bias]");
    //   print_tensor<float>(residual.get_float_ptr(),
    //                       32,
    //                       "[AddBiasResidualLayerNorm:forward:residual]");
    //   print_tensor<float>(added_output.get_float_ptr(),
    //                       32,
    //                       "[AddBiasResidualLayerNorm:forward:added_output]");
    //   print_tensor<float>(output.get_float_ptr(),
    //                       32,
    //                       "[AddBiasResidualLayerNorm:forward:output]");
    //   print_tensor<float>(gamma.get_float_ptr(),
    //                       32,
    //                       "[AddBiasResidualLayerNorm:forward:gamma]");
    //   print_tensor<float>(
    //       beta.get_float_ptr(), 32,
    //       "[AddBiasResidualLayerNorm:forward:beta]");
    // } else {
    //   print_tensor<half>(
    //       input.get_half_ptr(), 32,
    //       "[AddBiasResidualLayerNorm:forward:input]");
    //   print_tensor<half>(attn_bias.get_half_ptr(),
    //                      32,
    //                      "[AddBiasResidualLayerNorm:forward:attn_bias]");
    //   print_tensor<half>(residual.get_half_ptr(),
    //                      32,
    //                      "[AddBiasResidualLayerNorm:forward:residual]");
    //   print_tensor<half>(added_output.get_half_ptr(),
    //                      32,
    //                      "[AddBiasResidualLayerNorm:forward:added_output]");
    //   print_tensor<half>(output.get_half_ptr(),
    //                      32,
    //                      "[AddBiasResidualLayerNorm:forward:output]");
    //   print_tensor<half>(
    //       gamma.get_half_ptr(), 32,
    //       "[AddBiasResidualLayerNorm:forward:gamma]");
    //   print_tensor<half>(
    //       beta.get_half_ptr(), 32,
    //       "[AddBiasResidualLayerNorm:forward:beta]");
    // }
    // print_tensor<T>(in_ptr, 32, "[AddBiasResidualLayerNorm:forward:input]");
    // print_tensor<T>(out_ptr, 32,
    // "[AddBiasResidualLayerNorm:forward:output]");
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
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>((int)N);
    const T_ACC a = (db[index] * static_cast<T_ACC>(mean[index]) - ds[index]) *
                    static_cast<T_ACC>(rstd[index]) *
                    static_cast<T_ACC>(rstd[index]) *
                    static_cast<T_ACC>(rstd[index]) * s;
    c1[index] = a;
    c2[index] = -(a * static_cast<T_ACC>(mean[index]) +
                  db[index] * static_cast<T_ACC>(rstd[index]) * s);
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

template <typename T>
__device__ __inline__ void compute_gI(T const *__restrict__ dY,
                                      T const *__restrict__ X,
                                      T const *__restrict__ mean,
                                      T const *__restrict__ rstd,
                                      T const *__restrict__ gamma,
                                      T *dX,
                                      T *dX_residual,
                                      bool reset_input_grad,
                                      bool reset_residual_grad,
                                      int const N,
                                      T *buf) {
  auto const i1 = blockIdx.x;
  const T mean_val = mean[i1];
  const T rstd_val = rstd[i1];
  T stats_x1{0}, stats_x2{0};
  constexpr int unroll = 4;
  auto l = unroll * threadIdx.x;
  T const *X_i = X + i1 * N;
  T const *dY_i = dY + i1 * N;
  T *dX_i = dX + i1 * N;
  T *dX_residual_i = dX_residual + i1 * N;
  // vectorized reads don't improve perf, so use regular unrolling

  for (; l + unroll - 1 < N; l += blockDim.x * unroll) {
#pragma unroll
    for (int k = 0; k < unroll; k++) {
      T gamma_val = (gamma != nullptr) ? static_cast<T>(gamma[l + k]) : T(1);
      const T c_h = static_cast<T>(X_i[l + k]);
      const T c_loss = static_cast<T>(dY_i[l + k]);
      stats_x1 += c_loss * gamma_val;
      stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
    }
  }
  for (; l < N; l++) {
    T gamma_val = (gamma != nullptr) ? static_cast<T>(gamma[l]) : T(1);
    const T c_h = static_cast<T>(X_i[l]);
    const T c_loss = static_cast<T>(dY_i[l]);
    stats_x1 += c_loss * gamma_val;
    stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
  }

  stats_x1 = BlockReduceSum(stats_x1, buf);
  stats_x2 = BlockReduceSum(stats_x2, buf);
  if (threadIdx.x == 0) {
    buf[0] = stats_x1;
    buf[1] = stats_x2;
  }
  __syncthreads();
  stats_x1 = buf[0];
  stats_x2 = buf[1];
  T fH = N;
  T term1 = (T(1) / fH) * rstd_val;

  for (int l = threadIdx.x; l < N; l += blockDim.x) {
    const T x = X_i[l];
    const T dy = dY_i[l];
    T gamma_val = (gamma != nullptr) ? static_cast<T>(gamma[l]) : T(1);
    T f_grad_input = fH * gamma_val * dy;
    f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
    f_grad_input -= stats_x1;
    f_grad_input *= term1;
    if (reset_input_grad) {
      dX_i[l] = f_grad_input;
    } else {
      dX_i[l] += f_grad_input;
    }
    if (reset_residual_grad) {
      dX_residual_i[l] = f_grad_input;
    } else {
      dX_residual_i[l] += f_grad_input;
    }
  }
}

template <typename T>
__global__ void layer_norm_grad_input_kernel(T const *__restrict__ dY,
                                             T const *__restrict__ X,
                                             T const *__restrict__ mean,
                                             T const *__restrict__ rstd,
                                             T const *__restrict__ gamma,
                                             T *dX,
                                             T *dX_residual,
                                             bool reset_input_grad,
                                             bool reset_residual_grad,
                                             int const N) {
  alignas(sizeof(double)) extern __shared__ char s_data1[];
  T *buf = reinterpret_cast<T *>(&s_data1);

  compute_gI(dY,
             X,
             mean,
             rstd,
             gamma,
             dX,
             dX_residual,
             reset_input_grad,
             reset_residual_grad,
             N,
             buf);
}

/*static*/
template <typename T>
void AddBiasResidualLayerNorm::backward_kernel(
    AddBiasResidualLayerNormMeta const *m,
    T const *output_grad_ptr,
    T const *added_output_ptr,
    T *input_grad_ptr,
    T *residual_grad_ptr,
    T *attn_bias_grad_ptr,
    T const *gamma_ptr,
    T *gamma_grad_ptr,
    T *beta_grad_ptr,
    cudaStream_t stream) {
  const int64_t M = m->effective_batch_size;
  const int64_t N = m->effective_num_elements;
  ComputeInternalGradientsCUDAKernel<T>
      <<<M, kCUDABlockReduceNumThreads, 0, stream>>>(
          N,
          output_grad_ptr,
          added_output_ptr,
          gamma_ptr,
          static_cast<T *>(m->ds_ptr),
          static_cast<T *>(m->db_ptr));
  const int64_t B = (M + kCUDANumThreads - 1) / kCUDANumThreads;
  ComputeGradientFusedParamsCUDAKernel<T>
      <<<B, kCUDANumThreads, 0, stream>>>(M,
                                          N,
                                          static_cast<T *>(m->mean_ptr),
                                          static_cast<T *>(m->rstd_ptr),
                                          static_cast<T *>(m->ds_ptr),
                                          static_cast<T *>(m->db_ptr),
                                          static_cast<T *>(m->scale_ptr),
                                          static_cast<T *>(m->bias_ptr));
  int const warp_size = C10_WARP_SIZE;
  int const num_threads = 128;
  const dim3 blocks(M);
  int nshared = (num_threads / warp_size) * sizeof(T);
  layer_norm_grad_input_kernel<<<blocks, num_threads, nshared, stream>>>(
      output_grad_ptr,
      added_output_ptr,
      static_cast<T *>(m->mean_ptr),
      static_cast<T *>(m->rstd_ptr),
      gamma_ptr,
      input_grad_ptr,
      residual_grad_ptr,
      m->reset_input_grads[0],
      m->reset_input_grads[1],
      N);

  if (gamma_grad_ptr != NULL || beta_grad_ptr != NULL) {
    if (M < 512) {
      // For small batch size, do colwise reduce directly
      const int64_t B = (N + kCUDANumThreads - 1) / kCUDANumThreads;
      GammaBetaBackwardSimpleCUDAKernel<T>
          <<<B, kCUDANumThreads, 0, stream>>>(M,
                                              N,
                                              output_grad_ptr,
                                              added_output_ptr,
                                              static_cast<T *>(m->mean_ptr),
                                              static_cast<T *>(m->rstd_ptr),
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
              added_output_ptr,
              static_cast<T *>(m->mean_ptr),
              static_cast<T *>(m->rstd_ptr),
              gamma_grad_ptr,
              beta_grad_ptr);
    }
  }
}

/*static*/
void AddBiasResidualLayerNorm::backward_kernel_wrapper(
    AddBiasResidualLayerNormMeta const *m,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR &added_output,
    GenericTensorAccessorW &input_grad,
    GenericTensorAccessorW const &residual_grad,
    GenericTensorAccessorW const &attn_bias_grad,
    GenericTensorAccessorR const &gamma,
    GenericTensorAccessorW const &gamma_grad,
    GenericTensorAccessorW const &beta_grad) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  if (m->output_type[0] == DT_FLOAT) {
    AddBiasResidualLayerNorm::backward_kernel(
        m,
        output_grad.get_float_ptr(),
        added_output.get_float_ptr(),
        input_grad.get_float_ptr(),
        residual_grad.get_float_ptr(),
        attn_bias_grad.get_float_ptr(),
        m->elementwise_affine ? gamma.get_float_ptr() : nullptr,
        m->elementwise_affine ? gamma_grad.get_float_ptr() : nullptr,
        (m->elementwise_affine && m->use_bias) ? beta_grad.get_float_ptr()
                                               : nullptr,
        stream);
  } else if (m->output_type[0] == DT_HALF) {
    AddBiasResidualLayerNorm::backward_kernel(
        m,
        output_grad.get_half_ptr(),
        added_output.get_half_ptr(),
        input_grad.get_half_ptr(),
        residual_grad.get_half_ptr(),
        attn_bias_grad.get_half_ptr(),
        m->elementwise_affine ? gamma.get_half_ptr() : nullptr,
        m->elementwise_affine ? gamma_grad.get_half_ptr() : nullptr,
        (m->elementwise_affine && m->use_bias) ? beta_grad.get_half_ptr()
                                               : nullptr,
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
    printf("[AddBiasResidualLayerNorm] backward time (CF) = %.2fms\n", elapsed);
  }
}

/*static*/
template <typename T>
void AddBiasResidualLayerNorm::peft_bwd_kernel(
    AddBiasResidualLayerNormMeta const *m,
    T const *output_grad_ptr,
    T *input_grad_ptr,
    T *residual_grad_ptr,
    T const *gamma_ptr,
    cudaStream_t stream) {
  const int64_t M = m->effective_batch_size;
  const int64_t N = m->effective_num_elements;

  int const warp_size = C10_WARP_SIZE;
  int const num_threads = 128;
  const dim3 blocks(M);
  int nshared = (num_threads / warp_size) * sizeof(T);
  layer_norm_grad_input_kernel<<<blocks, num_threads, nshared, stream>>>(
      output_grad_ptr,
      static_cast<T const *>(m->input_activation),
      static_cast<T *>(m->mean_ptr),
      static_cast<T *>(m->rstd_ptr),
      gamma_ptr,
      input_grad_ptr,
      residual_grad_ptr,
      m->reset_input_grads[0],
      m->reset_input_grads[1],
      N);
}

/*static*/
void AddBiasResidualLayerNorm::peft_bwd_kernel_wrapper(
    AddBiasResidualLayerNormMeta const *m,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorW &input_grad,
    GenericTensorAccessorW const &residual_grad,
    GenericTensorAccessorR const &gamma) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  if (m->output_type[0] == DT_FLOAT) {
    peft_bwd_kernel(m,
                    output_grad.get_float_ptr(),
                    input_grad.get_float_ptr(),
                    residual_grad.get_float_ptr(),
                    m->elementwise_affine ? gamma.get_float_ptr() : nullptr,
                    stream);
  } else if (m->output_type[0] == DT_HALF) {
    peft_bwd_kernel(m,
                    output_grad.get_half_ptr(),
                    input_grad.get_half_ptr(),
                    residual_grad.get_half_ptr(),
                    m->elementwise_affine ? gamma.get_half_ptr() : nullptr,
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
    printf("[AddBiasResidualLayerNorm] peft_bwd time (CF) = %.2fms\n", elapsed);
  }
}

}; // namespace FlexFlow

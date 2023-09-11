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

AddBiasResidualLayerNormMeta::AddBiasResidualLayerNormMeta(
    FFHandler handle,
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
  size_t totalSize = effective_batch_size * data_type_size(data_type) * 3;
  gpu_mem_allocator.create_legion_instance(reserveInst, totalSize);
  mean_ptr = gpu_mem_allocator.allocate_instance_untyped(
      data_type_size(data_type) * effective_batch_size);
  rstd_ptr = gpu_mem_allocator.allocate_instance_untyped(
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
__global__ void LayerNormFusedForwardKernel(int attn_bias_dim,
                                            int residual_volume,
                                            int64_t effective_num_elements,
                                            float eps,
                                            T const *input_ptr,
                                            T const *attn_bias_ptr,
                                            T const *residual_ptr,
                                            T *output_ptr,
                                            T const *gamma_ptr,
                                            T const *beta_ptr,
                                            T *mean,
                                            T *rstd) {
  // Add attention bias and residual
  CUDA_KERNEL_LOOP(i, residual_volume) {
    int bias_idx = i % attn_bias_dim;
    output_ptr[i] = input_ptr[i] + attn_bias_ptr[bias_idx] + residual_ptr[i];
  }

  __syncthreads();

  // LayerNorm

  __shared__ float m_shared[C10_WARP_SIZE];
  __shared__ float v_shared[C10_WARP_SIZE];
  const int64_t i = blockIdx.x;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  for (int64_t j = threadIdx.x; j < effective_num_elements; j += blockDim.x) {
    const int64_t index = i * effective_num_elements + j;
    sum1 += static_cast<float>(output_ptr[index]);
    sum2 += static_cast<float>(output_ptr[index]) *
            static_cast<float>(output_ptr[index]);
  }
  sum1 = BlockReduceSum<float>(sum1, m_shared);
  sum2 = BlockReduceSum<float>(sum2, v_shared);
  if (threadIdx.x == 0) {
    float const scale = float(1) / static_cast<float>(effective_num_elements);
    sum1 *= scale;
    sum2 = max(sum2 * scale - sum1 * sum1, float(0));
    mean[i] = static_cast<T>(sum1);
    rstd[i] = static_cast<T>(rsqrt(sum2 + eps));
  }

  __syncthreads();

  using T_ACC = T;
  for (int64_t j = threadIdx.x; j < effective_num_elements; j += blockDim.x) {
    const int64_t index = i * effective_num_elements + j;
    const T_ACC gamma_v =
        gamma_ptr == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma_ptr[j]);
    const T_ACC beta_v =
        beta_ptr == nullptr ? T_ACC(0) : static_cast<T_ACC>(beta_ptr[j]);
    output_ptr[index] =
        (static_cast<T_ACC>(output_ptr[index]) - static_cast<T_ACC>(mean[i])) *
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
    T *output_ptr,
    T const *gamma_ptr,
    T const *beta_ptr,
    cudaStream_t stream) {

  int layer_norm_num_blocks = m->effective_batch_size;
  int layer_norm_num_threads =
      std::max(kCUDABlockReduceNumThreads, kCUDANumThreads);
  assert(layer_norm_num_threads <= CUDA_NUM_THREADS);
  int residual_num_blocks = GET_BLOCKS(residual_volume);
  int num_blocks = std::max(residual_num_blocks, layer_norm_num_blocks);
  int num_threads = std::max(layer_norm_num_threads,
                             std::min(residual_volume, CUDA_NUM_THREADS));

  LayerNormFusedForwardKernel<T>
      <<<num_blocks, num_threads, 0, stream>>>(attn_bias_dim,
                                               residual_volume,
                                               m->effective_num_elements,
                                               m->eps,
                                               input_ptr,
                                               attn_bias_ptr,
                                               residual_ptr,
                                               output_ptr,
                                               gamma_ptr,
                                               beta_ptr,
                                               static_cast<T *>(m->mean_ptr),
                                               static_cast<T *>(m->rstd_ptr));
}

/*static*/
void AddBiasResidualLayerNorm::inference_kernel_wrapper(
    AddBiasResidualLayerNormMeta const *m,
    int attn_bias_dim,
    int residual_volume,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW &output,
    GenericTensorAccessorR const &residual,
    GenericTensorAccessorR const &attn_bias,
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
    AddBiasResidualLayerNorm::inference_kernel<float>(
        m,
        attn_bias_dim,
        residual_volume,
        input.get_float_ptr(),
        attn_bias.get_float_ptr(),
        residual.get_float_ptr(),
        output.get_float_ptr(),
        gamma.get_float_ptr(),
        m->use_bias ? beta.get_float_ptr() : nullptr,
        stream);
  } else if (m->input_type[0] == DT_HALF) {
    AddBiasResidualLayerNorm::inference_kernel<half>(
        m,
        attn_bias_dim,
        residual_volume,
        input.get_half_ptr(),
        attn_bias.get_half_ptr(),
        residual.get_half_ptr(),
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
    printf("[AddBiasResidualLayerNorm] forward time (CF) = %.9fms\n", elapsed);
    // print_tensor<T>(in_ptr, 32, "[AddBiasResidualLayerNorm:forward:input]");
    // print_tensor<T>(out_ptr, 32,
    // "[AddBiasResidualLayerNorm:forward:output]");
  }
}

}; // namespace FlexFlow

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

#include "flexflow/ops/kernels/rms_norm_kernels.h"
#include "flexflow/ops/rms_norm.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;
using Legion::Domain;

RMSNormMeta::RMSNormMeta(FFHandler handler, RMSNorm const *rms)
    : OpMeta(handler, gather) {
  eps = rms->eps;
  // fixme
  checkCUDA(cudaMalloc(&mean_ptr, sizeof(float) * 1000));
}

namespace Kernels {
namespace RMSNorm {
template <typename T>
void forward_kernel_wrapper(LayerNormMeta const *m,
                            T const *in_ptr,
                            T *out_ptr,
                            T *gamma_ptr,
                            T *beta_ptr) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  Internal::forward_kernel(input.get(),
                           output.get_half_ptr(),
                           weight.get_half_ptr(),
                           in_dim,
                           out_dim,
                           batch_size,
                           m->aggr,
                           output.domain.get_volume(),
                           stream);
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[LayerNorm] forward time (CF) = %.2fms\n", elapsed);
    print_tensor<T>(in_ptr, 32, "[LayerNorm:forward:input]");
    print_tensor<T>(out_ptr, 32, "[LayerNorm:forward:output]");
  }
}

namespace Internal {
/*static*/
template <typename T>
void LayerNorm::forward_kernel(RMSNormMeta const *m,
                               T const *in_ptr,
                               T *out_ptr,
                               T *gamma_ptr,
                               T *beta_ptr,
                               cudaStream_t stream) {
  RowwiseMomentsCUDAKernel<float>
      <<<m->effective_batch_size, kCUDABlockReduceNumThreads, 0, stream>>>(
          m->effective_num_elements, m->eps, in_ptr, m->mean_ptr, m->rstd_ptr);
  LayerNormForwardCUDAKernel<float>
      <<<m->effective_batch_size, kCUDANumThreads, 0, stream>>>(
          m->effective_num_elements,
          in_ptr,
          m->mean_ptr,
          m->rstd_ptr,
          gamma_ptr,
          beta_ptr,
          out_ptr);
  cuApplyLayerNorm_<float>

template<typename T, typename U, typename V> __device__
void cuApplyLayerNorm_(
  V* __restrict__ output_vals,
  U* __restrict__ mean,
  U* __restrict__ invvar,
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const U epsilon,
  const V* __restrict__ gamma,
  const V* __restrict__ beta,
  bool rms_only
  )
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensors are contiguous
  //
  for (auto i1=blockIdx.y; i1 < n1; i1 += gridDim.y) {
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    U mu,sigma2;
    cuWelfordMuSigma2(vals,n1,n2,i1,mu,sigma2,buf,rms_only);

    const T* lvals = vals + i1*n2;
    V* ovals = output_vals + i1*n2;
    U c_invvar = rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL && (beta != NULL || rms_only)) {
      for (int i = thrx;  i < n2;  i+=numx) {
        U curr = static_cast<U>(lvals[i]);
        if (!rms_only) {
          ovals[i] = gamma[i] * static_cast<V>(c_invvar * (curr - mu)) + beta[i];
        } else {
          ovals[i] = gamma[i] * static_cast<V>(c_invvar * curr);
        }

      }
    } else {
      for (int i = thrx;  i < n2;  i+=numx) {
        U curr = static_cast<U>(lvals[i]);
        if (!rms_only) {
          ovals[i] = static_cast<V>(c_invvar * (curr - mu));
        } else {
          ovals[i] = static_cast<V>(c_invvar * curr);
        }
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      if (!rms_only) {
        mean[i1] = mu;
      }
      invvar[i1] = c_invvar;
    }
    __syncthreads();
  }
}
}
} // namespace Internal
// namespace Internal
} // namespace RMSNorm
} // namespace Kernels

}; // namespace FlexFlow
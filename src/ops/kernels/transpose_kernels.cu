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

#include "flexflow/ops/kernels/transpose_kernels.h"
#include "flexflow/ops/transpose.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;
using Legion::Domain;

TransposeMeta::TransposeMeta(FFHandler handler, Transpose const *transpose)
    : OpMeta(handler, transpose) {}

struct TransposeStrides {
  int num_dim;
  int in_strides[MAX_TENSOR_DIM], out_strides[MAX_TENSOR_DIM],
      perm[MAX_TENSOR_DIM];
};

namespace Kernels {
namespace Transpose {

void forward_kernel_wrapper(TransposeMeta const *m,
                            float const *input_ptr,
                            float *output_ptr,
                            Domain in_domain,
                            Domain out_domain) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  Internal::forward_kernel(
      m, input_ptr, output_ptr, in_domain, out_domain, stream);
}

void backward_kernel_wrapper(TransposeMeta const *m,
                             float *input_grad_ptr,
                             float const *output_grad_ptr,
                             Domain in_grad_domain,
                             Domain out_grad_domain) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  Internal::backward_kernel(m,
                            input_grad_ptr,
                            output_grad_ptr,
                            in_grad_domain,
                            out_grad_domain,
                            stream);
}

namespace Internal {

__global__ void transpose_simple_kernel(coord_t volume,
                                        float const *in_ptr,
                                        float *out_ptr,
                                        const TransposeStrides info,
                                        float const beta) {
  CUDA_KERNEL_LOOP(o_idx, volume) {
    coord_t i_idx = 0;
    coord_t t = o_idx;
    for (int i = info.num_dim - 1; i >= 0; i--) {
      coord_t ratio = t / info.out_strides[i];
      t -= ratio * info.out_strides[i];
      i_idx += ratio * info.in_strides[info.perm[i]];
    }
    out_ptr[o_idx] += out_ptr[o_idx] * beta + in_ptr[i_idx];
  }
}

void forward_kernel(TransposeMeta const *m,
                    float const *input_ptr,
                    float *output_ptr,
                    Domain in_domain,
                    Domain out_domain,
                    cudaStream_t stream) {
  TransposeStrides info;
  info.num_dim = out_domain.get_dim();
  assert(info.num_dim == m->num_dim);
  for (int i = 0; i < info.num_dim; i++) {
    if (i == 0) {
      info.in_strides[i] = 1;
      info.out_strides[i] = 1;
    } else {
      int in_dim_size = (in_domain.hi()[i - 1] - in_domain.lo()[i - 1] + 1);
      int out_dim_size = (out_domain.hi()[i - 1] - out_domain.lo()[i - 1] + 1);
      info.in_strides[i] = info.in_strides[i - 1] * in_dim_size;
      info.out_strides[i] = info.out_strides[i - 1] * out_dim_size;
    }
    info.perm[i] = m->perm[i];
  }
  transpose_simple_kernel<<<GET_BLOCKS(out_domain.get_volume()),
                            CUDA_NUM_THREADS,
                            0,
                            stream>>>(
      out_domain.get_volume(), input_ptr, output_ptr, info, 0.0f /*beta*/);
}

void backward_kernel(TransposeMeta const *m,
                     float *input_grad_ptr,
                     float const *output_grad_ptr,
                     Domain in_grad_domain,
                     Domain out_grad_domain,
                     cudaStream_t stream) {
  TransposeStrides info;
  info.num_dim = in_grad_domain.get_dim();
  assert(info.num_dim == m->num_dim);
  for (int i = 0; i < info.num_dim; i++) {
    if (i == 0) {
      info.in_strides[i] = 1;
      info.out_strides[i] = 1;
    } else {
      int in_dim_size =
          (out_grad_domain.hi()[i - 1] - out_grad_domain.lo()[i - 1] + 1);
      int out_dim_size =
          (in_grad_domain.hi()[i - 1] - in_grad_domain.lo()[i - 1] + 1);
      info.in_strides[i] = info.in_strides[i - 1] * in_dim_size;
      info.out_strides[i] = info.out_strides[i - 1] * out_dim_size;
    }
    info.perm[m->perm[i]] = i;
  }
  transpose_simple_kernel<<<GET_BLOCKS(in_grad_domain.get_volume()),
                            CUDA_NUM_THREADS,
                            0,
                            stream>>>(in_grad_domain.get_volume(),
                                      output_grad_ptr,
                                      input_grad_ptr,
                                      info,
                                      1.0f /*beta*/);
}

} // namespace Internal
} // namespace Transpose
} // namespace Kernels
} // namespace FlexFlow

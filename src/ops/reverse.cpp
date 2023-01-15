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

#include "flexflow/ops/reverse.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;

__global__ void reverse_forward_kernel(float const *in_ptr,
                                       float *out_ptr,
                                       coord_t num_out_blks,
                                       coord_t reverse_dim_size,
                                       coord_t in_blk_size) {
  CUDA_KERNEL_LOOP(i, num_out_blks * reverse_dim_size * in_blk_size) {
    coord_t blk_idx = i / (reverse_dim_size * in_blk_size);
    i = i - blk_idx * (reverse_dim_size * in_blk_size);
    coord_t reverse_dim_idx = i / in_blk_size;
    i = i - reverse_dim_idx * in_blk_size;
    coord_t in_idx = blk_idx * (reverse_dim_size * in_blk_size) +
                     (reverse_dim_size - 1 - reverse_dim_idx) * in_blk_size + i;
    out_ptr[i] = in_ptr[in_idx];
  }
}

/*static*/
void Reverse::forward_kernel(float const *in_ptr,
                             float *out_ptr,
                             coord_t num_out_blks,
                             coord_t reverse_dim_size,
                             coord_t in_blk_size,
                             coord_t output_size,
                             hipStream_t stream) {
  hipLaunchKernelGGL(reverse_forward_kernel,
                     GET_BLOCKS(output_size),
                     CUDA_NUM_THREADS,
                     0,
                     stream,
                     in_ptr,
                     out_ptr,
                     num_out_blks,
                     reverse_dim_size,
                     in_blk_size);
}

/*static*/
void Reverse::forward_kernel_wrapper(float const *in_ptr,
                                     float *out_ptr,
                                     coord_t num_out_blks,
                                     coord_t reverse_dim_size,
                                     coord_t in_blk_size,
                                     coord_t output_size) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  Reverse::forward_kernel(in_ptr,
                          out_ptr,
                          num_out_blks,
                          reverse_dim_size,
                          in_blk_size,
                          output_size,
                          stream);
}

/*static*/
void Reverse::backward_kernel(float const *out_grad_ptr,
                              float *in_grad_ptr,
                              coord_t num_out_blks,
                              coord_t reverse_dim_size,
                              coord_t in_blk_size,
                              coord_t input_size,
                              hipStream_t stream) {
  hipLaunchKernelGGL(reverse_forward_kernel,
                     GET_BLOCKS(input_size),
                     CUDA_NUM_THREADS,
                     0,
                     stream,
                     out_grad_ptr,
                     in_grad_ptr,
                     num_out_blks,
                     reverse_dim_size,
                     in_blk_size);
}

/*static*/
void Reverse::backward_kernel_wrapper(float const *out_grad_ptr,
                                      float *in_grad_ptr,
                                      coord_t num_out_blks,
                                      coord_t reverse_dim_size,
                                      coord_t in_blk_size,
                                      coord_t input_size) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  Reverse::backward_kernel(out_grad_ptr,
                           in_grad_ptr,
                           num_out_blks,
                           reverse_dim_size,
                           in_blk_size,
                           input_size,
                           stream);
}

}; // namespace FlexFlow

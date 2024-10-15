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

#include "device.h"
#include "kernels/reverse_kernels.h"

namespace FlexFlow {

namespace Kernels {
namespace Reverse {

// __global__ void reverse_forward_kernel(float const *in_ptr,
//                                        float *out_ptr,
//                                        coord_t num_out_blks,
//                                        coord_t reverse_dim_size,
//                                        coord_t in_blk_size) {
//   CUDA_KERNEL_LOOP(i, num_out_blks * reverse_dim_size * in_blk_size) {
//     coord_t out_idx = i;
//     coord_t blk_idx = i / (reverse_dim_size * in_blk_size);
//     i = i - blk_idx * (reverse_dim_size * in_blk_size);
//     coord_t reverse_dim_idx = i / in_blk_size;
//     i = i - reverse_dim_idx * in_blk_size;
//     coord_t in_idx = blk_idx * (reverse_dim_size * in_blk_size) +
//                      (reverse_dim_size - 1 - reverse_dim_idx) * in_blk_size +
//                      i;
//     out_ptr[out_idx] = in_ptr[in_idx];
//   }
// CUDA_KERNEL_LOOP(i, num_out_blks * reverse_dim_size * in_blk_size) {
//   coord_t blk_idx = i / (reverse_dim_size * in_blk_size);
//   i = i - blk_idx * (reverse_dim_size * in_blk_size);
//   coord_t reverse_dim_idx = i / in_blk_size;
//   i = i - reverse_dim_idx * in_blk_size;
//   coord_t in_idx = blk_idx * (reverse_dim_size * in_blk_size) +
//                    (reverse_dim_size - 1 - reverse_dim_idx) * in_blk_size +
//                    i;
//   out_ptr[i] = in_ptr[in_idx];
// }
// }

/* I mentioned this earlier, but I still think the reverse_forward_kernel code
   is incorrect, even though it matches the code in inference/master? Whenever
   I'm testing the code and printing out the output, I'm getting unexpected
   outputs, and I think it's a result of modifying the loop index i in the
   previous code?
*/
__global__ void reverse_forward_kernel(float const *in_ptr,
                                       float *out_ptr,
                                       coord_t num_out_blks,
                                       coord_t reverse_dim_size,
                                       coord_t in_blk_size) {
  CUDA_KERNEL_LOOP(i, num_out_blks * reverse_dim_size * in_blk_size) {
    coord_t blk_idx = i / (reverse_dim_size * in_blk_size);
    coord_t idx_within_blk = i % (reverse_dim_size * in_blk_size);
    coord_t reverse_dim_idx = idx_within_blk / in_blk_size;
    coord_t in_idx = idx_within_blk % in_blk_size;
    coord_t input_index =
        blk_idx * (reverse_dim_size * in_blk_size) +
        (reverse_dim_size - 1 - reverse_dim_idx) * in_blk_size + in_idx;
    out_ptr[i] = in_ptr[input_index];
  }
}

void forward_kernel(cudaStream_t stream,
                    float const *in_ptr,
                    float *out_ptr,
                    coord_t num_out_blks,
                    coord_t reverse_dim_size,
                    coord_t in_blk_size,
                    coord_t output_size) {

  reverse_forward_kernel<<<GET_BLOCKS(output_size),
                           CUDA_NUM_THREADS,
                           0,
                           stream>>>(
      in_ptr, out_ptr, num_out_blks, reverse_dim_size, in_blk_size);
}

void backward_kernel(cudaStream_t stream,
                     float const *out_grad_ptr,
                     float *in_grad_ptr,
                     coord_t num_out_blks,
                     coord_t reverse_dim_size,
                     coord_t in_blk_size,
                     coord_t input_size) {

  reverse_forward_kernel<<<GET_BLOCKS(input_size),
                           CUDA_NUM_THREADS,
                           0,
                           stream>>>(
      out_grad_ptr, in_grad_ptr, num_out_blks, reverse_dim_size, in_blk_size);
}

} // namespace Reverse
} // namespace Kernels
} // namespace FlexFlow

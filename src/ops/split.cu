/* Copyright 2020 Facebook
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

#include "flexflow/ops/split.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;

/*static*/
void Split::forward_kernel(float **out_ptrs,
                           float const *in_ptr,
                           coord_t const *out_blk_sizes,
                           coord_t in_blk_size,
                           coord_t num_blks,
                           int numOutputs,
                           cudaStream_t stream) {
  for (int i = 0; i < numOutputs; i++) {
    copy_with_stride<<<GET_BLOCKS(out_blk_sizes[i] * num_blks),
                       CUDA_NUM_THREADS,
                       0,
                       stream>>>(
        out_ptrs[i], in_ptr, num_blks, out_blk_sizes[i], in_blk_size);
    in_ptr += out_blk_sizes[i];
  }
}

/*static*/
void Split::forward_kernel_wrapper(float **out_ptrs,
                                   float const *in_ptr,
                                   coord_t const *out_blk_sizes,
                                   coord_t in_blk_size,
                                   coord_t num_blks,
                                   int numOutputs) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  Split::forward_kernel(out_ptrs,
                        in_ptr,
                        out_blk_sizes,
                        in_blk_size,
                        num_blks,
                        numOutputs,
                        stream);
}

/*static*/
void Split::backward_kernel(float *in_grad_ptr,
                            float const **out_grad_ptr,
                            coord_t const *out_blk_sizes,
                            coord_t in_blk_size,
                            coord_t num_blks,
                            int numOutputs,
                            cudaStream_t stream) {
  for (int i = 0; i < numOutputs; i++) {
    add_with_stride<<<GET_BLOCKS(out_blk_sizes[i] * num_blks),
                      CUDA_NUM_THREADS,
                      0,
                      stream>>>(
        in_grad_ptr, out_grad_ptr[i], num_blks, in_blk_size, out_blk_sizes[i]);
    in_grad_ptr += out_blk_sizes[i];
  }
}

/*static*/
void Split::backward_kernel_wrapper(float *in_grad_ptr,
                                    float const **out_grad_ptr,
                                    coord_t const *out_blk_sizes,
                                    coord_t in_blk_size,
                                    coord_t num_blks,
                                    int numOutputs) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  Split::backward_kernel(in_grad_ptr,
                         out_grad_ptr,
                         out_blk_sizes,
                         in_blk_size,
                         num_blks,
                         numOutputs,
                         stream);
  // checkCUDA(cudaDeviceSynchronize());
}

}; // namespace FlexFlow

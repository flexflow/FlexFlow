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

#include "internal/device.h"
#include "kernels/concat_kernels_gpu.h"
#include <cassert>

namespace FlexFlow {

static void calc_blk_size(size_t &num_blocks,
                          size_t &blk_size,
                          TensorShape const &shape,
                          ff_dim_t axis) {
  blk_size = get_num_elements(slice_tensor_dims(shape.dims, axis, std::nullopt))
                 .int_from_positive_int();
  num_blocks =
      get_num_elements(slice_tensor_dims(shape.dims, ff_dim_t{0_n}, axis))
          .int_from_positive_int();
}

void concat_gpu_forward_kernel(
    cudaStream_t stream,
    GenericTensorAccessorW const &output,
    std::vector<GenericTensorAccessorR> const &inputs,
    ff_dim_t axis) {
  assert(inputs.size() <= MAX_NUM_INPUTS);
  size_t num_blocks = 1, output_blk_size = 1;
  calc_blk_size(num_blocks, output_blk_size, output.shape, axis);
  off_t offset = 0;

  for (GenericTensorAccessorR const &input : inputs) {
    size_t input_num_blocks = 1, input_blk_size = 1;
    calc_blk_size(input_num_blocks, input_blk_size, input.shape, axis);
    assert(input_num_blocks == num_blocks || output_blk_size == input_blk_size);

    int blocks_to_copy =
        (output_blk_size == input_blk_size) ? input_num_blocks : num_blocks;

    copy_with_stride<<<GET_BLOCKS(input_blk_size * num_blocks),
                       CUDA_NUM_THREADS,
                       0,
                       stream>>>(output.get_float_ptr() + offset,
                                 input.get_float_ptr(),
                                 blocks_to_copy,
                                 output_blk_size,
                                 input_blk_size);

    offset += (output_blk_size == input_blk_size)
                  ? input_blk_size * input_num_blocks
                  : input_blk_size;
  }
}

void concat_gpu_backward_kernel(
    cudaStream_t stream,
    GenericTensorAccessorR const &output_grad,
    std::vector<GenericTensorAccessorW> const &input_grads,
    ff_dim_t axis) {
  assert(input_grads.size() <= MAX_NUM_INPUTS);
  size_t num_blocks = 1, output_blk_size = 1;
  calc_blk_size(num_blocks, output_blk_size, output_grad.shape, axis);
  off_t offset = 0;

  for (auto &input_grad : input_grads) {
    size_t input_num_blocks = 1, input_blk_size = 1;
    calc_blk_size(input_num_blocks, input_blk_size, input_grad.shape, axis);
    assert(input_num_blocks == num_blocks || output_blk_size == input_blk_size);

    int blocks_to_add =
        (output_blk_size == input_blk_size) ? input_num_blocks : num_blocks;

    add_with_stride<<<GET_BLOCKS(input_blk_size * num_blocks),
                      CUDA_NUM_THREADS,
                      0,
                      stream>>>(input_grad.get_float_ptr(),
                                output_grad.get_float_ptr() + offset,
                                blocks_to_add,
                                input_blk_size,
                                output_blk_size);

    offset += (output_blk_size == input_blk_size)
                  ? input_blk_size * input_num_blocks
                  : input_blk_size;
  }
}

} // namespace FlexFlow

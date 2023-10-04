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
#include "kernels/concat_kernels.h"
#include "kernels/device.h"
#include <cassert>

namespace FlexFlow {
namespace Kernels {
namespace Concat {

void calc_blk_size(size_t &num_blocks,
                   size_t &blk_size,
                   ArrayShape const &shape,
                   req<ff_dim_t> legion_axis) {
  num_blocks = 1;
  blk_size = 1;
  for (int d = 0; d < shape.num_dims(); d++) {
    if (d <= legion_axis) {
      blk_size *= shape[legion_dim_t(d)];
    } else {
      num_blocks *= shape[legion_dim_t(d)];
    }
  }
}

void forward_kernel(ffStream_t stream,
                    GenericTensorAccessorW const &output,
                    std::vector<FlexFlow::GenericTensorAccessorR> const &inputs,
                    int num_inputs,
                    ff_dim_t legion_axis) {
  size_t num_blocks = 1, output_blk_size = 1, input_blk_sizes[MAX_NUM_INPUTS];
  assert(num_inputs <= MAX_NUM_INPUTS);
  calc_blk_size(num_blocks, output_blk_size, output.shape, legion_axis);
  for (int i = 0; i < num_inputs; i++) {
    size_t input_num_blocks = 1;
    calc_blk_size(
        input_num_blocks, input_blk_sizes[i], inputs[i].shape, legion_axis);
    assert(input_num_blocks == num_blocks);
  }

  off_t offset = 0;
  for (int i = 0; i < num_inputs; i++) {
    copy_with_stride<<<GET_BLOCKS(input_blk_sizes[i] * num_blocks),
                       CUDA_NUM_THREADS,
                       0,
                       stream>>>(output.get_float_ptr() + offset,
                                 inputs[i].get_float_ptr(),
                                 num_blocks,
                                 output_blk_size,
                                 input_blk_sizes[i]);
    offset += input_blk_sizes[i];
  }
}

void backward_kernel(
    ffStream_t stream,
    GenericTensorAccessorR const &output_grad,
    std::vector<FlexFlow::GenericTensorAccessorW> const &input_grads,
    int num_inputs,
    ff_dim_t legion_axis) {
  size_t num_blocks = 1, output_blk_size = 1, input_blk_sizes[MAX_NUM_INPUTS];
  assert(num_inputs <= MAX_NUM_INPUTS);

  ArrayShape shape = output_grad.shape;
  calc_blk_size(num_blocks, output_blk_size, shape, legion_axis);
  for (int i = 0; i < num_inputs; i++) {
    shape = input_grads[i].shape;
    size_t input_num_blocks = 1;
    calc_blk_size(input_num_blocks, input_blk_sizes[i], shape, legion_axis);
    assert(input_num_blocks == num_blocks);
  }

  off_t offset = 0;
  for (int i = 0; i < num_inputs; i++) {
    add_with_stride<<<GET_BLOCKS(input_blk_sizes[i] * num_blocks),
                      CUDA_NUM_THREADS,
                      0,
                      stream>>>(input_grads[i].get_float_ptr(),
                                output_grad.get_float_ptr() + offset,
                                num_blocks,
                                input_blk_sizes[i],
                                output_blk_size);
    offset += input_blk_sizes[i];
  }
}

} // namespace Concat
} // namespace Kernels
} // namespace FlexFlow

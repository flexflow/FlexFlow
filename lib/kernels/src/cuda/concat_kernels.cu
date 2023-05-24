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

#include "kernels/concat_kernels.h"
#include "kernels/cuda_helper.h"
#include <cassert>

namespace FlexFlow {

namespace Kernels {
namespace Concat {

void init_meta(ConcatPerDeviceState *m, int legion_axis) {
  m->legion_axis = legion_axis;
}


void calc_blk_size(size_t &num_blocks,
                   size_t &blk_size,
                   ArrayShape const &shape,
                   int axis) {
  num_blocks = 1;
  blk_size = 1;
  for (int d = 0; d < shape.num_dims(); d++) {
    if (d <= axis) {
      blk_size *= shape[d];
    } else {
      num_blocks *= shape[d];
    }
  }
}

void forward_kernel(cudaStream_t stream,
                    ConcatPerDeviceState const *m,
                    GenericTensorAccessorW const &output,
                    GenericTensorAccessorR const *inputs,
                    int num_inputs) {
  size_t num_blocks = 1, output_blk_size = 1, input_blk_sizes[MAX_NUM_INPUTS];
  assert(num_inputs <= MAX_NUM_INPUTS);
  calc_blk_size(num_blocks, output_blk_size, output.shape, m->legion_axis);
  for (int i = 0; i < num_input; i++) {
    size_t input_num_blocks = 1;
    calc_blk_size(input_num_blocks, input_blk_sizes[i], inputs[i].shape, m->legion_axis);
    assert (input_num_blocks == num_blocks);
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
    // printf("output = %x num_blocks=%d output_blk_size=%d
    // input_blk_size[%d]=%d\n",
    //        output, num_blocks, output_blk_size, i, input_blk_sizes[i]);
    offset += input_blk_sizes[i];
  }
}

void backward_kernel(cudaStream_t stream,
                     ConcatPerDeviceState const *m,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorW const *input_grads,
                     int num_inputs) {
  size_t num_blocks = 1, output_blk_size = 1, input_blk_sizes[MAX_NUM_INPUTS];
  assert(num_inputs <= MAX_NUM_INPUTS);
  switch (output_grad.domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    Rect<DIM> rect = output_grad.domain;                                       \
    calc_blk_size<DIM>(num_blocks, output_blk_size, rect, m->legion_axis);               \
    for (int i = 0; i < num_inputs; i++) {                                     \
      rect = input_grads[i].domain;                                            \
      coord_t input_num_blocks = 1;                                            \
      calc_blk_size<DIM>(input_num_blocks, input_blk_sizes[i], rect, m->legion_axis);    \
      assert(input_num_blocks == num_blocks);                                  \
    }                                                                          \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      fprintf(stderr, "Unsupported concat dimension number");
      assert(false);
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

  // Rect<2> output_rect(Point<2>(0, 0), Point<2>(output_blk_size-1, batch_size
  // - 1)); Rect<2> input_rect(Point<2>(0, 0), Point<2>(input_blk_sizes[0]-1,
  // batch_size - 1)); print_tensor<2, float>(output_grad - output_blk_size,
  // output_rect, "[Concat:backward:output]"); print_tensor<2,
  // float>(input_grads[0], input_rect, "[Concat:backward:input0]");
}

} // namespace Concat
} // namespace Kernels
} // namespace FlexFlow

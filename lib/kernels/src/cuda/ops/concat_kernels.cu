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
#include "op-attrs/tensor_dims.h"

namespace FlexFlow {

// The number of contiguous elements spanned by `axis` and everything after it.
static int get_blk_size(TensorShape const &shape, ff_dim_t axis) {
  return get_num_elements(slice_tensor_dims(shape.dims, axis, std::nullopt))
      .int_from_positive_int();
}

// The number of such blocks, i.e. the product of the dims before `axis`.
static int get_num_blks(TensorShape const &shape, ff_dim_t axis) {
  return get_num_elements(slice_tensor_dims(shape.dims, ff_dim_t{0_n}, axis))
      .int_from_positive_int();
}

void concat_gpu_forward_kernel(
    cudaStream_t stream,
    ConcatAttrs const &attrs,
    std::vector<GenericTensorAccessorR> const &inputs,
    GenericTensorAccessorW const &output) {
  ASSERT(inputs.size() == attrs.num_inputs.int_from_int_ge_two());
  ASSERT(inputs.size() <= MAX_NUM_INPUTS);

  int num_blks = get_num_blks(output.shape, attrs.axis);
  int output_blk_size = get_blk_size(output.shape, attrs.axis);

  int offset = 0;
  for (GenericTensorAccessorR const &input : inputs) {
    ASSERT(get_num_blks(input.shape, attrs.axis) == num_blks);
    int input_blk_size = get_blk_size(input.shape, attrs.axis);

    copy_with_stride<<<GET_BLOCKS(input_blk_size * num_blks),
                       CUDA_NUM_THREADS,
                       0,
                       stream>>>(output.get_float_ptr() + offset,
                                 input.get_float_ptr(),
                                 num_blks,
                                 output_blk_size,
                                 input_blk_size);

    offset += input_blk_size;
  }

  ASSERT(offset == output_blk_size);
}

void concat_gpu_backward_kernel(
    cudaStream_t stream,
    ConcatAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    std::vector<GenericTensorAccessorR> const &inputs,
    std::vector<GenericTensorAccessorW> const &input_grads) {
  ASSERT(input_grads.size() == attrs.num_inputs.int_from_int_ge_two());
  ASSERT(input_grads.size() <= MAX_NUM_INPUTS);

  int num_blks = get_num_blks(output_grad.shape, attrs.axis);
  int output_blk_size = get_blk_size(output_grad.shape, attrs.axis);

  int offset = 0;
  for (GenericTensorAccessorW const &input_grad : input_grads) {
    ASSERT(get_num_blks(input_grad.shape, attrs.axis) == num_blks);
    int input_blk_size = get_blk_size(input_grad.shape, attrs.axis);

    add_with_stride<<<GET_BLOCKS(input_blk_size * num_blks),
                      CUDA_NUM_THREADS,
                      0,
                      stream>>>(input_grad.get_float_ptr(),
                                output_grad.get_float_ptr() + offset,
                                num_blks,
                                input_blk_size,
                                output_blk_size);

    offset += input_blk_size;
  }

  ASSERT(offset == output_blk_size);
}

} // namespace FlexFlow

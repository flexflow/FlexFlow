#include "kernels/split_kernels.h"
#include "kernels/split_kernels_cpu.h"
#include "kernels/split_kernels_gpu.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

static std::pair<positive_int, positive_int>
    calc_block_size(TensorShape const &tensor_shape, ff_dim_t axis) {
  positive_int num_blocks = 1_p;
  positive_int block_size = 1_p;
  for (nonnegative_int d :
       nonnegative_range(get_num_elements(tensor_shape.dims)
                             .nonnegative_int_from_positive_int())) {
    if (d <= axis.value) {
      block_size *= dim_at_idx(tensor_shape.dims, legion_dim_t{d});
    } else {
      num_blocks *= dim_at_idx(tensor_shape.dims, legion_dim_t{d});
    }
  }
  return {num_blocks, block_size};
}

void split_forward_kernel(device_stream_t const &stream,
                          SplitAttrs const &attrs,
                          GenericTensorAccessorR const &input,
                          std::vector<GenericTensorAccessorW> const &outputs) {
  if (stream.is_gpu()) {
    int out_block_sizes[MAX_NUM_OUTPUTS];
    auto [num_blocks, in_block_size] = calc_block_size(input.shape, attrs.axis);

    for (int i = 0; i < attrs.splits.size(); i++) {
      auto [_, out_block_size] =
          calc_block_size(outputs.at(i).shape, attrs.axis);
      out_block_sizes[i] = out_block_size.int_from_positive_int();
    }

    std::vector<float *> output_ptrs =
        transform(outputs, [&](GenericTensorAccessorW const &t) -> float * {
          return t.get_float_ptr();
        });

    split_gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*out_ptrs=*/output_ptrs.data(),
        /*in_ptr=*/input.get_float_ptr(),
        /*out_blk_sizes=*/out_block_sizes,
        /*in_blk_size=*/in_block_size.int_from_positive_int(),
        /*num_blks=*/num_blocks.int_from_positive_int(),
        /*numOutputs=*/attrs.splits.size());
  } else {
    split_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*outputs=*/outputs);
  }
}

void split_backward_kernel(
    device_stream_t const &stream,
    SplitAttrs const &attrs,
    std::vector<GenericTensorAccessorR> const &output_grads,
    GenericTensorAccessorW const &input_grad) {
  if (stream.is_gpu()) {
    int out_block_sizes[MAX_NUM_OUTPUTS];
    auto [num_blocks, in_block_size] =
        calc_block_size(input_grad.shape, attrs.axis);

    for (int i = 0; i < attrs.splits.size(); i++) {
      int out_num_blocks;
      auto [_, out_block_size] =
          calc_block_size(output_grads.at(i).shape, attrs.axis);
      out_block_sizes[i] = out_block_size.int_from_positive_int();
    }

    std::vector<float const *> output_grad_ptrs = transform(
        output_grads, [&](GenericTensorAccessorR const &t) -> float const * {
          return t.get_float_ptr();
        });

    split_gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*in_grad_ptr=*/input_grad.get_float_ptr(),
        /*out_grad_ptr=*/output_grad_ptrs.data(),
        /*out_blk_sizes=*/out_block_sizes,
        /*in_blk_size=*/in_block_size.int_from_positive_int(),
        /*num_blks=*/num_blocks.int_from_positive_int(),
        /*numOutputs=*/attrs.splits.size());
  } else {
    ASSERT(stream.is_cpu());
    split_cpu_backward_kernel(
        /*attrs=*/attrs,
        /*output_grads=*/output_grads,
        /*input_grad=*/input_grad);
  }
}

} // namespace FlexFlow

#include "kernels/reverse_kernels_cpu.h"
#include <algorithm>
#include <vector>

namespace FlexFlow {
namespace Kernels {
namespace Reverse {

void cpu_reverse_forward_kernel(float const *in_ptr,
                                float *out_ptr,
                                coord_t num_out_blks,
                                coord_t reverse_dim_size,
                                coord_t in_blk_size) {
  coord_t total_elements = num_out_blks * reverse_dim_size * in_blk_size;

  std::vector<std::vector<float>> in_blocks(num_out_blks * reverse_dim_size,
                                            std::vector<float>(in_blk_size));

  // For each output block, copy the input block into in_blocks
  for (coord_t blk_idx = 0; blk_idx < num_out_blks; ++blk_idx) {
    // Each output block has reverse_dim_size input blocks
    for (coord_t rev_idx = 0; rev_idx < reverse_dim_size; ++rev_idx) {
      coord_t start_idx = (blk_idx * reverse_dim_size + rev_idx) * in_blk_size;

      // Copy elements from in_ptr to the current block in in_blocks
      std::vector<float> &current_block =
          in_blocks[blk_idx * reverse_dim_size + rev_idx];
      for (coord_t i = 0; i < in_blk_size; ++i) {
        current_block[i] = in_ptr[start_idx + i];
      }
    }
  }

  // Reverse the in_blocks within each output block
  for (coord_t blk_idx = 0; blk_idx < num_out_blks; ++blk_idx) {
    auto block_start = in_blocks.begin() + blk_idx * reverse_dim_size;
    auto block_end = block_start + reverse_dim_size;
    std::reverse(block_start, block_end);
  }

  // Copy the reversed blocks to the output array
  for (coord_t blk_idx = 0; blk_idx < num_out_blks; ++blk_idx) {
    for (coord_t rev_idx = 0; rev_idx < reverse_dim_size; ++rev_idx) {
      coord_t start_idx = (blk_idx * reverse_dim_size + rev_idx) * in_blk_size;

      // Copy elements from the current block in in_blocks to out_ptr
      std::vector<float> const &current_block =
          in_blocks[blk_idx * reverse_dim_size + rev_idx];
      for (coord_t i = 0; i < in_blk_size; ++i) {
        out_ptr[start_idx + i] = current_block[i];
      }
    }
  }
}

void cpu_forward_kernel(float const *in_ptr,
                        float *out_ptr,
                        coord_t num_out_blks,
                        coord_t reverse_dim_size,
                        coord_t in_blk_size,
                        coord_t output_size) {
  cpu_reverse_forward_kernel(
      in_ptr, out_ptr, num_out_blks, reverse_dim_size, in_blk_size);
}

void cpu_backward_kernel(float const *out_grad_ptr,
                         float *in_grad_ptr,
                         coord_t num_out_blks,
                         coord_t reverse_dim_size,
                         coord_t in_blk_size,
                         coord_t input_size) {
  cpu_reverse_forward_kernel(
      out_grad_ptr, in_grad_ptr, num_out_blks, reverse_dim_size, in_blk_size);
}

} // namespace Reverse
} // namespace Kernels
} // namespace FlexFlow

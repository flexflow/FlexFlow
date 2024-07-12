#include "kernels/reverse_kernels_cpu.h"
#include <iostream>

namespace FlexFlow {
namespace Kernels {
namespace Reverse {
namespace CPU {

void reverse_forward_kernel(float const *in_ptr,
                            float *out_ptr,
                            coord_t num_out_blks,
                            coord_t reverse_dim_size,
                            coord_t in_blk_size) {
  coord_t total_elements = num_out_blks * reverse_dim_size * in_blk_size;
  for (coord_t i = 0; i < total_elements; ++i) {
    coord_t blk_idx = i / (reverse_dim_size * in_blk_size);
    coord_t offset = i - blk_idx * (reverse_dim_size * in_blk_size);
    coord_t reverse_dim_idx = offset / in_blk_size;
    coord_t in_idx = blk_idx * (reverse_dim_size * in_blk_size) +
                     (reverse_dim_size - 1 - reverse_dim_idx) * in_blk_size +
                     (offset % in_blk_size);
    out_ptr[i] = in_ptr[in_idx];
  }
}

void forward_kernel(float const *in_ptr,
                    float *out_ptr,
                    coord_t num_out_blks,
                    coord_t reverse_dim_size,
                    coord_t in_blk_size,
                    coord_t output_size) {
  reverse_forward_kernel(
      in_ptr, out_ptr, num_out_blks, reverse_dim_size, in_blk_size);
}

void backward_kernel(float const *out_grad_ptr,
                     float *in_grad_ptr,
                     coord_t num_out_blks,
                     coord_t reverse_dim_size,
                     coord_t in_blk_size,
                     coord_t input_size) {
  reverse_forward_kernel(
      out_grad_ptr, in_grad_ptr, num_out_blks, reverse_dim_size, in_blk_size);
}

} // namespace CPU
} // namespace Reverse
} // namespace Kernels
} // namespace FlexFlow

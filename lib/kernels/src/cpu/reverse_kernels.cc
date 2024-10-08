#include "kernels/reverse_kernels_cpu.h"
#include "kernels/datatype_dispatch.h"
#include <algorithm>
#include <vector>

namespace FlexFlow::Kernels::Reverse {

template <DataType DT>
struct CPUReverseForwardKernel {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW &output,
                  coord_t num_out_blks,
                  coord_t reverse_dim_size,
                  coord_t in_blk_size) {
    assert(input.data_type == DT && output.data_type == DT);

    // For each output block, copy the input block
    for (coord_t blk_idx = 0; blk_idx < num_out_blks; ++blk_idx) {
      for (coord_t rev_idx = 0; rev_idx < reverse_dim_size; ++rev_idx) {
        for (coord_t i = 0; i < in_blk_size; ++i) {
          output.at<DT>(blk_idx, rev_idx, i) =
              input.at<DT>(blk_idx, rev_idx, i);
        }
      }
    }

    // Reverse the blocks within each output block
    for (coord_t blk_idx = 0; blk_idx < num_out_blks; ++blk_idx) {
      for (coord_t rev_idx = 0; rev_idx < reverse_dim_size / 2; ++rev_idx) {
        coord_t start_idx = rev_idx;
        coord_t end_idx = reverse_dim_size - 1 - rev_idx;

        for (coord_t i = 0; i < in_blk_size; ++i) {
          std::swap(output.at<DT>(blk_idx, start_idx, i),
                    output.at<DT>(blk_idx, end_idx, i));
        }
      }
    }
  }
};

void cpu_forward_kernel(GenericTensorAccessorR const &input_accessor,
                        GenericTensorAccessorW &output_accessor,
                        coord_t num_out_blks,
                        coord_t reverse_dim_size,
                        coord_t in_blk_size) {
  DataTypeDispatch1<CPUReverseForwardKernel>{}(input_accessor.data_type,
                                               input_accessor,
                                               std::ref(output_accessor),
                                               num_out_blks,
                                               reverse_dim_size,
                                               in_blk_size);
}

void cpu_backward_kernel(GenericTensorAccessorR const &output_accessor,
                         GenericTensorAccessorW &input_accessor,
                         coord_t num_out_blks,
                         coord_t reverse_dim_size,
                         coord_t in_blk_size) {
  DataTypeDispatch1<CPUReverseForwardKernel>{}(output_accessor.data_type,
                                               output_accessor,
                                               std::ref(input_accessor),
                                               num_out_blks,
                                               reverse_dim_size,
                                               in_blk_size);
}

} // namespace FlexFlow::Kernels::Reverse

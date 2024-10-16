#include "kernels/datatype_dispatch.h"
#include "kernels/reverse_kernels_cpu.h"
#include <algorithm>
#include <vector>

namespace FlexFlow::Kernels::Reverse {

template <DataType DT>
struct CPUReverseForwardKernel {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW &output) {
    assert(input.data_type == DT && output.data_type == DT);

    coord_t num_out_blocks = input.shape.at(legion_dim_t(0));
    coord_t reverse_dim_size = input.shape.at(legion_dim_t(1));
    coord_t in_block_size = input.shape.at(legion_dim_t(2));

    for (coord_t block_idx = 0; block_idx < num_out_blocks; block_idx++) {
      for (coord_t rev_idx = 0; rev_idx < reverse_dim_size; rev_idx++) {
        for (coord_t i = 0; i < in_block_size; i++) {
          output.at<DT>(block_idx, rev_idx, i) =
              input.at<DT>(num_out_blocks - 1 - block_idx,
                           reverse_dim_size - 1 - rev_idx,
                           in_block_size - 1 - i);
        }
      }
    }
  }
};

void cpu_forward_kernel(GenericTensorAccessorR const &input_accessor,
                        GenericTensorAccessorW &output_accessor) {
  DataTypeDispatch1<CPUReverseForwardKernel>{}(
      input_accessor.data_type, input_accessor, std::ref(output_accessor));
}

void cpu_backward_kernel(GenericTensorAccessorR const &output_accessor,
                         GenericTensorAccessorW &input_accessor) {
  DataTypeDispatch1<CPUReverseForwardKernel>{}(
      output_accessor.data_type, output_accessor, std::ref(input_accessor));
}

} // namespace FlexFlow::Kernels::Reverse

#include "kernels/concat_kernels_cpu.h"
#include "kernels/accessor.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/tensor_accessor_binary_ops.h"
#include "kernels/tensor_accessor_split.h"
#include "utils/containers/foldl1.h"

namespace FlexFlow {

void concat_cpu_forward_kernel(
    GenericTensorAccessorW const &output,
    std::vector<GenericTensorAccessorR> const &inputs,
    ff_dim_t axis) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();

  GenericTensorAccessorR result = foldl1(
      inputs,
      [&](GenericTensorAccessorR const &accum,
          GenericTensorAccessorR const &elem) -> GenericTensorAccessorR {
        return tensor_accessor_binary_concat(accum, elem, axis, cpu_allocator);
      });

  copy_accessor_data_to_l_from_r(
      /*dst_accessor=*/output,
      /*src_accessor=*/result);
}

void concat_cpu_backward_kernel(
    GenericTensorAccessorR const &output_grad,
    std::vector<GenericTensorAccessorW> const &input_grads,
    ff_dim_t axis) {

  std::vector<positive_int> input_grad_axis_sizes =
      transform(input_grads,
                [&](GenericTensorAccessorW const &input_grad) -> positive_int {
                  return dim_at_idx(input_grad.shape.dims, axis);
                });

  tensor_accessor_split_to(
      /*input=*/output_grad,
      /*axis=*/axis,
      /*sizes=*/input_grad_axis_sizes,
      /*outputs=*/input_grads);
}

} // namespace FlexFlow

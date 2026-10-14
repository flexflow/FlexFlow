#include "op-attrs/ops/flat.h"
#include "op-attrs/ff_ordered/ff_ordered_concat.h"
#include "op-attrs/ff_ordered/ff_ordered_slice.h"
#include "op-attrs/ff_ordered/ff_ordered_slice_inclusive.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/all_of.h"
#include "utils/containers/product.h"
#include <cassert>

namespace FlexFlow {

TensorShape flat_get_output_shape(FlatAttrs const &attrs,
                                  TensorShape const &input_shape) {
  FFOrdered<positive_int> leading_dims = ff_ordered_slice(
      ff_ordered(input_shape.dims), ff_dim_t{0_n}, attrs.start_dim);
  FFOrdered<positive_int> flattened_dims = ff_ordered_slice_inclusive(
      ff_ordered(input_shape.dims), attrs.start_dim, attrs.end_dim);
  FFOrdered<positive_int> trailing_dims =
      ff_ordered_slice(ff_ordered(input_shape.dims),
                       add_to_ff_dim(attrs.end_dim, 1),
                       std::nullopt);

  if (flattened_dims.empty()) {
    return input_shape;
  }

  return TensorShape{
      TensorDims{
          ff_ordered_concat(std::vector{
              leading_dims,
              FFOrdered{product(flattened_dims)},
              trailing_dims,
          }),
      },
      input_shape.data_type,
  };
}

ParallelTensorDimDegrees flat_get_output_parallel_dim_degrees(
    FlatAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) {
  FFOrdered<positive_int> flattened_dim_degrees = ff_ordered_slice_inclusive(
      input_degrees.shard_degrees, attrs.start_dim, attrs.end_dim);

  if (flattened_dim_degrees.empty()) {
    return input_degrees;
  }

  ASSERT(all_of(flattened_dim_degrees,
                [](positive_int degree) { return degree == 1; }),
         "flat_get_output_parallel_dim_degrees expected all shard degrees of "
         "flattened dimensions to be 1",
         attrs,
         input_degrees,
         flattened_dim_degrees);

  return ParallelTensorDimDegrees{
      /*sum_degree=*/input_degrees.sum_degree,
      /*discard_copy_degree=*/input_degrees.discard_copy_degree,
      /*shard_degrees=*/
      ff_ordered_concat(std::vector{
          ff_ordered_slice(
              input_degrees.shard_degrees, ff_dim_t{0_n}, attrs.start_dim),
          FFOrdered{product(flattened_dim_degrees)},
          ff_ordered_slice(input_degrees.shard_degrees,
                           add_to_ff_dim(attrs.end_dim, 1),
                           std::nullopt),
      }),
  };
}

ParallelTensorShape
    flat_get_output_parallel_shape(FlatAttrs const &attrs,
                                   ParallelTensorShape const &input_shape) {
  TensorShape unpar =
      flat_get_output_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees degrees = flat_get_output_parallel_dim_degrees(
      attrs, get_parallel_degrees(input_shape));

  return lift_shape_to_parallel_with_degrees(unpar, degrees);
}

} // namespace FlexFlow

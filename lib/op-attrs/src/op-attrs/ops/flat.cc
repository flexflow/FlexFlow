#include "op-attrs/ops/flat.h"
#include "op-attrs/dim_ordered/concat.h"
#include "op-attrs/dim_ordered/slice.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/any_of.h"
#include "utils/containers/product.h"
#include "op-attrs/dim_ordered/slice.h"
#include <cassert>

namespace FlexFlow {

TensorShape get_output_shape(FlatAttrs const &attrs, TensorShape const &input_shape) {
  FFOrdered<size_t> leading_dims = slice(ff_ordered(input_shape.dims), ff_dim_t{0}, attrs.start_dim);
  FFOrdered<size_t> flattened_dims = slice(ff_ordered(input_shape.dims), attrs.start_dim, attrs.end_dim);
  FFOrdered<size_t> trailing_dims = slice(ff_ordered(input_shape.dims), attrs.end_dim, std::nullopt);

  if (flattened_dims.empty()) {
    return input_shape;
  }

  return TensorShape{
    TensorDims{
      concat(std::vector{
        leading_dims,
        {product(flattened_dims)},
        trailing_dims,
      }),
    },
    input_shape.data_type,
  };
}

tl::expected<ParallelTensorDimDegrees, std::string> get_output_parallel_dim_degrees(FlatAttrs const &attrs,
                                                                                    ParallelTensorDimDegrees const &input_degrees) {
  FFOrdered<int> flattened_dim_degrees = slice(input_degrees.shard_degrees, attrs.start_dim, attrs.end_dim);

  if (flattened_dim_degrees.empty()) {
    return input_degrees;
  }

  if (any_of(flattened_dim_degrees, [](int degree) { return degree != 1; })) {
    return tl::unexpected(fmt::format("get_output_parallel_dim_degrees for {} expected all shard degrees of flattened dimensions to be 1, but received {}", attrs, input_degrees));
  }

  return ParallelTensorDimDegrees{
    /*sum_degree=*/input_degrees.sum_degree,
    /*discard_copy_degree=*/input_degrees.discard_copy_degree,
    /*shard_degrees=*/concat(std::vector{
      slice(input_degrees.shard_degrees, ff_dim_t{0}, attrs.start_dim),
      {product(flattened_dim_degrees)},
      slice(input_degrees.shard_degrees, attrs.end_dim, std::nullopt),
    }),
  };
}

tl::expected<ParallelTensorShape, std::string> get_output_shape(FlatAttrs const &attrs,
                                                                ParallelTensorShape const &input_shape) {
  TensorShape unpar = get_output_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees degrees = ({
    tl::expected<ParallelTensorDimDegrees, std::string> returned = get_output_parallel_dim_degrees(attrs, get_parallel_degrees(input_shape));
    if (!returned.has_value()) {
      return tl::unexpected(returned.error());
    }
    returned.value();
  });

  return lift_to_parallel_with_degrees(unpar, degrees);
}

} // namespace FlexFlow

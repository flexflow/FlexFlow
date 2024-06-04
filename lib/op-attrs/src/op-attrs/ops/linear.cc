#include "op-attrs/ops/linear.h"
#include "op-attrs/dim_ordered/slice.h"
#include "op-attrs/dim_ordered/transform.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {

tl::expected<TensorShape, std::string>
    get_kernel_shape(LinearAttrs const &attrs, TensorShape const &input_shape) {
  size_t in_channels = dim_at_idx(input_shape, ff_dim_t{-1});

  return TensorShape{
      TensorDims{
          FFOrdered<size_t>{in_channels, size_t_from_int(attrs.out_channels)},
      },
      input_shape.data_type,
  };
}

tl::expected<TensorShape, std::string>
    get_bias_shape(LinearAttrs const &attrs, TensorShape const &input_shape) {
  return TensorShape{
      TensorDims{
          FFOrdered<size_t>{size_t_from_int(attrs.out_channels)},
      },
      input_shape.data_type,
  };
}

tl::expected<TensorShape, std::string>
    get_output_shape(LinearAttrs const &attrs, TensorShape const &input_shape) {
  TensorShape output_shape = input_shape;
  output_shape.dims.ff_ordered.at(ff_dim_t{-1}) =
      size_t_from_int(attrs.out_channels);

  return output_shape;
}

tl::expected<ParallelTensorShape, std::string>
    get_kernel_shape(LinearAttrs const &attrs,
                     ParallelTensorShape const &input) {
  TensorShape unpar = ({
    tl::expected<TensorShape, std::string> result_unpar =
        get_kernel_shape(attrs, get_reduced_shape(input));
    if (!result_unpar.has_value()) {
      return tl::unexpected(result_unpar.error());
    }
    result_unpar.value();
  });

  SumDegree sum_degree = 1;
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{
      get_sum_degree(input) *
      product(
          slice(ff_ordered_shard_degrees(input), std::nullopt, ff_dim_t{-1}))};
  FFOrdered<int> shard_degrees = FFOrdered<int>{
      shard_dim_at_idx(input, ff_dim_t{-1}).degree,
      get_discard_copy_degree(input),
  };

  return lift_to_parallel_with_degrees(
      unpar, sum_degree, discard_copy_degree, shard_degrees);
}

tl::expected<ParallelTensorShape, std::string>
    get_bias_shape(LinearAttrs const &attrs, ParallelTensorShape const &input) {
  TensorShape unpar = ({
    tl::expected<TensorShape, std::string> result_unpar =
        get_bias_shape(attrs, get_reduced_shape(input));
    if (!result_unpar.has_value()) {
      return tl::unexpected(result_unpar.error());
    }
    result_unpar.value();
  });

  SumDegree sum_degree =
      get_sum_degree(input) * shard_dim_at_idx(input, ff_dim_t{-1}).degree;
  DiscardCopyDegree discard_copy_degree = product(
      slice(ff_ordered_shard_degrees(input), std::nullopt, ff_dim_t{-1}));
  FFOrdered<int> shard_degrees = FFOrdered<int>{get_discard_copy_degree(input)};

  return lift_to_parallel_with_degrees(
      unpar, sum_degree, discard_copy_degree, shard_degrees);
}

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(LinearAttrs const &attrs,
                     ParallelTensorShape const &input) {
  TensorShape unpar = ({
    tl::expected<TensorShape, std::string> result_unpar =
        get_output_shape(attrs, get_reduced_shape(input));
    if (!result_unpar.has_value()) {
      return tl::unexpected(result_unpar.error());
    }
    result_unpar.value();
  });

  SumDegree sum_degree =
      get_sum_degree(input) * shard_dim_at_idx(input, ff_dim_t{-1}).degree;
  DiscardCopyDegree discard_copy_degree = 1;
  FFOrdered<int> shard_degrees = ff_ordered_shard_degrees(input);
  shard_degrees.at(ff_dim_t{-1}) = get_discard_copy_degree(input);

  return lift_to_parallel_with_degrees(
      unpar, sum_degree, discard_copy_degree, shard_degrees);
}

} // namespace FlexFlow

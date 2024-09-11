#include "op-attrs/ops/pool_2d.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

tl::expected<TensorShape, std::string>
  get_output_shape(Pool2DAttrs const &attrs,
                             TensorShape const &input_shape) {
  if (num_dims(input_shape) != 4) {
    return tl::unexpected(fmt::format("get_output_shape for Pool2DAttrs expected input tensor to have 4 dims, but received shape {}", input_shape));
  }

  size_t num_samples = dim_at_idx(input_shape, ff_dim_t{0});
  size_t num_channels = dim_at_idx(input_shape, ff_dim_t{1});
  size_t input_height = dim_at_idx(input_shape, ff_dim_t{2});
  size_t input_width = dim_at_idx(input_shape, ff_dim_t{3});

  size_t output_height =
      (input_height + 2 * attrs.padding_h - attrs.kernel_h) / attrs.stride_h +
      1;

  size_t output_width =
      (input_width + 2 * attrs.padding_w - attrs.kernel_w) / attrs.stride_w + 1;

  return TensorShape{TensorDims{FFOrdered<size_t>{
                         num_samples,
                         num_channels,
                         output_height,
                         output_width,
                     }},
                     input_shape.data_type};
}

tl::expected<ParallelTensorShape, std::string>
  get_output_shape(Pool2DAttrs const &attrs, ParallelTensorShape const &input_shape) {
  TensorShape unpar = ({
    tl::expected<TensorShape, std::string> result_unpar =
        get_output_shape(attrs, get_reduced_shape(input_shape));
    if (!result_unpar.has_value()) {
      return tl::unexpected(result_unpar.error());
    }
    result_unpar.value();
  });

  ParallelTensorDimDegrees degrees = ({
    tl::expected<ParallelTensorDimDegrees, std::string> result_degrees =
        get_output_parallel_dim_degrees(attrs, get_parallel_degrees(input_shape));
    if (!result_degrees.has_value()) {
      return tl::unexpected(result_degrees.error());
    }
    result_degrees.value();
  });

  return lift_to_parallel_with_degrees(unpar, degrees);
}

tl::expected<ParallelTensorDimDegrees, std::string>
  get_output_parallel_dim_degrees(Pool2DAttrs const &attrs,
                   ParallelTensorDimDegrees const &input_degrees) {
  if (input_degrees.sum_degree.value > 1) {
    if (attrs.pool_type == PoolOp::MAX) {
      return tl::unexpected(fmt::format("get_output_parallel_dim_degrees for Pool2DAttrs with PoolOp::MAX expected input sum degree == 1, but received {}", input_degrees));
    } else if (attrs.activation.has_value()) { 
      return tl::unexpected(fmt::format("get_output_parallel_dim_degrees for Pool2DAttrs with activation={} expected input sum degree == 1, but received {}", attrs.activation.value(), input_degrees));
    }
  }
    
  return input_degrees;
}

} // namespace FlexFlow

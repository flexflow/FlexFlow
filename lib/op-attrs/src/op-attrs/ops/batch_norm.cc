#include "op-attrs/ops/batch_norm.h"
#include "op-attrs/dim_ordered/concat.h"
#include "op-attrs/dim_ordered/slice.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/any_of.h"
#include "utils/containers/extend.h"

namespace FlexFlow {

std::vector<IncomingTensorRole>
    get_batch_norm_incoming_tensor_roles(BatchNormAttrs const &attrs) {
  std::vector<IncomingTensorRole> result = {IncomingTensorRole::INPUT};

  if (attrs.affine) {
    extend(result,
           std::vector{IncomingTensorRole::WEIGHT, IncomingTensorRole::WEIGHT});
  }

  return result;
}

static std::optional<std::string>
    check_input_shape(BatchNormAttrs const &, TensorShape const &input_shape) {
  if (num_dims(input_shape) < 2) {
    return fmt::format(
        "BatchNormAttrs expected input dims >= 2, but received input shape {}",
        input_shape);
  }

  if (input_shape.data_type != DataType::FLOAT) {
    return fmt::format("BatchNormAttrs currently only supports data_type = "
                       "FLOAT, but received input data_type {}. "
                       "If you need this feature, please create an issue.",
                       input_shape.data_type);
  }

  return std::nullopt;
}

tl::expected<TensorShape, std::string>
    get_output_shape(BatchNormAttrs const &attrs,
                     TensorShape const &input_shape) {
  {
    std::optional<std::string> maybe_err_msg =
        check_input_shape(attrs, input_shape);
    if (maybe_err_msg.has_value()) {
      return tl::unexpected(maybe_err_msg.value());
    }
  }

  return input_shape;
}

tl::expected<TensorShape, std::string>
    get_gamma_weights_shape(BatchNormAttrs const &attrs,
                            TensorShape const &input_shape) {
  {
    std::optional<std::string> maybe_err_msg =
        check_input_shape(attrs, input_shape);
    if (maybe_err_msg.has_value()) {
      return tl::unexpected(maybe_err_msg.value());
    }
  }

  if (!attrs.affine) {
    return tl::unexpected("No gamma weights exist for attrs.affine = false");
  }

  size_t num_channels = dim_at_idx(input_shape, ff_dim_t{1});

  return TensorShape{
      TensorDims{FFOrdered<size_t>{
          num_channels,
      }},
      DataType::FLOAT,
  };
}

tl::expected<TensorShape, std::string>
    get_beta_weights_shape(BatchNormAttrs const &attrs,
                           TensorShape const &input_shape) {

  if (!attrs.affine) {
    return tl::unexpected("No beta weights exist for attrs.affine = false");
  }

  return get_gamma_weights_shape(attrs, input_shape);
}

static std::optional<std::string>
    check_input_degrees(BatchNormAttrs const &,
                        ParallelTensorDimDegrees const &input_degrees) {
  if (input_degrees.shard_degrees.size() < 2) {
    return fmt::format("BatchNormAttrs expected input dims >= 2, but received "
                       "input degrees {}",
                       input_degrees);
  }

  if (input_degrees.sum_degree != SumDegree{1}) {
    return fmt::format("Expected sum degree 1, but receieved sum degree {}",
                       input_degrees.sum_degree);
  }

  if (input_degrees.discard_copy_degree != DiscardCopyDegree{1}) {
    return fmt::format(
        "Expected discard copy degree 1, but receieved discard copy degree {}",
        input_degrees.discard_copy_degree);
  }

  FFOrdered<int> non_channel_degrees =
      concat(slice(input_degrees.shard_degrees, ff_dim_t{0}, ff_dim_t{1}),
             slice(input_degrees.shard_degrees, ff_dim_t{2}, std::nullopt));

  if (any_of(non_channel_degrees, [](int degree) { return degree != 1; })) {
    return fmt::format("Expected parallel degree of all non-channel dimensions "
                       "to be 1, but received input with degrees {}",
                       input_degrees);
  }

  return std::nullopt;
}

tl::expected<ParallelTensorDimDegrees, std::string>
    get_output_parallel_dim_degrees(
        BatchNormAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {
  {
    std::optional<std::string> maybe_err_msg =
        check_input_degrees(attrs, input_degrees);
    if (maybe_err_msg.has_value()) {
      return tl::unexpected(maybe_err_msg.value());
    }
  }

  return input_degrees;
}

tl::expected<ParallelTensorDimDegrees, std::string>
    get_gamma_weights_parallel_dim_degrees(
        BatchNormAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {
  {
    std::optional<std::string> maybe_err_msg =
        check_input_degrees(attrs, input_degrees);
    if (maybe_err_msg.has_value()) {
      return tl::unexpected(maybe_err_msg.value());
    }
  }

  if (!attrs.affine) {
    return tl::unexpected("No gamma weights exist for attrs.affine = false");
  }

  ff_dim_t channel_dim = ff_dim_t{1};

  return ParallelTensorDimDegrees{
      SumDegree{1},
      DiscardCopyDegree{1},
      FFOrdered<int>{input_degrees.shard_degrees.at(channel_dim)},
  };
}

tl::expected<ParallelTensorDimDegrees, std::string>
    get_beta_weights_parallel_dim_degrees(
        BatchNormAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {
  {
    std::optional<std::string> maybe_err_msg =
        check_input_degrees(attrs, input_degrees);
    if (maybe_err_msg.has_value()) {
      return tl::unexpected(maybe_err_msg.value());
    }
  }

  if (!attrs.affine) {
    return tl::unexpected("No beta weights exist for attrs.affine = false");
  }

  return get_gamma_weights_parallel_dim_degrees(attrs, input_degrees);
}

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(BatchNormAttrs const &attrs,
                     ParallelTensorShape const &input_shape) {
  TensorShape unpar = ({
    tl::expected<TensorShape, std::string> returned =
        get_output_shape(attrs, get_reduced_shape(input_shape));
    if (!returned.has_value()) {
      return tl::unexpected(returned.error());
    }
    returned.value();
  });

  ParallelTensorDimDegrees degrees = ({
    tl::expected<ParallelTensorDimDegrees, std::string> returned =
        get_output_parallel_dim_degrees(attrs,
                                        get_parallel_degrees(input_shape));
    if (!returned.has_value()) {
      return tl::unexpected(returned.error());
    }
    returned.value();
  });

  return lift_to_parallel_with_degrees(unpar, degrees);
}

tl::expected<ParallelTensorShape, std::string>
    get_gamma_weights_shape(BatchNormAttrs const &attrs,
                            ParallelTensorShape const &input_shape) {

  TensorShape unpar = ({
    tl::expected<TensorShape, std::string> returned =
        get_gamma_weights_shape(attrs, get_reduced_shape(input_shape));
    if (!returned.has_value()) {
      return tl::unexpected(returned.error());
    }
    returned.value();
  });

  ParallelTensorDimDegrees degrees = ({
    tl::expected<ParallelTensorDimDegrees, std::string> returned =
        get_gamma_weights_parallel_dim_degrees(
            attrs, get_parallel_degrees(input_shape));
    if (!returned.has_value()) {
      return tl::unexpected(returned.error());
    }
    returned.value();
  });

  return lift_to_parallel_with_degrees(unpar, degrees);
}

tl::expected<ParallelTensorShape, std::string>
    get_beta_weights_shape(BatchNormAttrs const &attrs,
                           ParallelTensorShape const &input_shape) {

  TensorShape unpar = ({
    tl::expected<TensorShape, std::string> returned =
        get_beta_weights_shape(attrs, get_reduced_shape(input_shape));
    if (!returned.has_value()) {
      return tl::unexpected(returned.error());
    }
    returned.value();
  });

  ParallelTensorDimDegrees degrees = ({
    tl::expected<ParallelTensorDimDegrees, std::string> returned =
        get_beta_weights_parallel_dim_degrees(
            attrs, get_parallel_degrees(input_shape));
    if (!returned.has_value()) {
      return tl::unexpected(returned.error());
    }
    returned.value();
  });

  return lift_to_parallel_with_degrees(unpar, degrees);
}

} // namespace FlexFlow

#include "op-attrs/ops/batch_norm.h"
#include "op-attrs/ff_ordered/ff_ordered_concat.h"
#include "op-attrs/ff_ordered/ff_ordered_slice.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/any_of.h"
#include "utils/containers/extend.h"
#include "utils/expected.h"

namespace FlexFlow {

std::map<TensorSlotName, IncomingTensorRole>
    get_batch_norm_incoming_tensor_roles(BatchNormAttrs const &attrs) {
  std::map<TensorSlotName, IncomingTensorRole> result = {
      {
          TensorSlotName::INPUT,
          IncomingTensorRole::INPUT,
      },
  };

  if (attrs.affine) {
    result[TensorSlotName::GAMMA] = IncomingTensorRole::WEIGHT;
    result[TensorSlotName::BETA] = IncomingTensorRole::WEIGHT;
  }

  return result;
}

static std::optional<std::string>
    check_input_shape(BatchNormAttrs const &, TensorShape const &input_shape) {
  if (get_num_dims(input_shape.dims) < 2) {
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

  positive_int num_channels =
      dim_at_idx(input_shape.dims, relative_ff_dim_t{1});

  return TensorShape{
      TensorDims{FFOrdered<positive_int>{
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

tl::expected<std::map<TensorSlotName, TensorShape>, std::string>
    get_weight_shapes(BatchNormAttrs const &attrs,
                      TensorShape const &input_shape) {

  TensorShape gamma_shape =
      PROPAGATE_ERR(get_gamma_weights_shape(attrs, input_shape));
  TensorShape beta_shape =
      PROPAGATE_ERR(get_beta_weights_shape(attrs, input_shape));

  return std::map<TensorSlotName, TensorShape>{
      {
          TensorSlotName::GAMMA,
          gamma_shape,
      },
      {
          TensorSlotName::BETA,
          beta_shape,
      },
  };
}

static std::optional<std::string>
    check_input_degrees(BatchNormAttrs const &,
                        ParallelTensorDimDegrees const &input_degrees) {
  if (input_degrees.shard_degrees.size() < 2) {
    return fmt::format("BatchNormAttrs expected input dims >= 2, but received "
                       "input degrees {}",
                       input_degrees);
  }

  if (input_degrees.sum_degree != SumDegree{1_p}) {
    return fmt::format("Expected sum degree 1, but receieved sum degree {}",
                       input_degrees.sum_degree);
  }

  if (input_degrees.discard_copy_degree != DiscardCopyDegree{1_p}) {
    return fmt::format(
        "Expected discard copy degree 1, but receieved discard copy degree {}",
        input_degrees.discard_copy_degree);
  }

  FFOrdered<positive_int> non_channel_degrees = ff_ordered_concat(
      ff_ordered_slice(
          input_degrees.shard_degrees, ff_dim_t{0_n}, ff_dim_t{1_n}),
      ff_ordered_slice(
          input_degrees.shard_degrees, ff_dim_t{2_n}, std::nullopt));

  if (any_of(non_channel_degrees,
             [](positive_int degree) { return degree != 1_p; })) {
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

  relative_ff_dim_t channel_dim = relative_ff_dim_t{1};

  return ParallelTensorDimDegrees{
      SumDegree{1_p},
      DiscardCopyDegree{1_p},
      FFOrdered<positive_int>{input_degrees.shard_degrees.at(channel_dim)},
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

tl::expected<std::map<TensorSlotName, ParallelTensorDimDegrees>, std::string>
    get_weight_parallel_dim_degrees(
        BatchNormAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {

  ParallelTensorDimDegrees gamma_degrees = PROPAGATE_ERR(
      get_gamma_weights_parallel_dim_degrees(attrs, input_degrees));
  ParallelTensorDimDegrees beta_degrees = PROPAGATE_ERR(
      get_beta_weights_parallel_dim_degrees(attrs, input_degrees));

  return std::map<TensorSlotName, ParallelTensorDimDegrees>{
      {
          TensorSlotName::GAMMA,
          gamma_degrees,
      },
      {
          TensorSlotName::BETA,
          beta_degrees,
      },
  };
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

  return lift_shape_to_parallel_with_degrees(unpar, degrees);
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

  return lift_shape_to_parallel_with_degrees(unpar, degrees);
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

  return lift_shape_to_parallel_with_degrees(unpar, degrees);
}

tl::expected<std::map<TensorSlotName, ParallelTensorShape>, std::string>
    get_weight_shapes(BatchNormAttrs const &attrs,
                      ParallelTensorShape const &input_shape) {

  ParallelTensorShape gamma_shape =
      PROPAGATE_ERR(get_gamma_weights_shape(attrs, input_shape));
  ParallelTensorShape beta_shape =
      PROPAGATE_ERR(get_beta_weights_shape(attrs, input_shape));

  return std::map<TensorSlotName, ParallelTensorShape>{
      {
          TensorSlotName::GAMMA,
          gamma_shape,
      },
      {
          TensorSlotName::BETA,
          beta_shape,
      },
  };
}

tl::expected<std::map<TensorSlotName, InitializerAttrs>, std::string>
    get_initializers(BatchNormAttrs const &attrs) {
  if (attrs.affine) {
    InitializerAttrs gamma_initializer =
        InitializerAttrs{ConstantInitializerAttrs{DataTypeValue{float{1}}}};

    InitializerAttrs beta_initializer =
        InitializerAttrs{ConstantInitializerAttrs{DataTypeValue{float{0}}}};

    return std::map<TensorSlotName, InitializerAttrs>{
        {
            TensorSlotName::GAMMA,
            gamma_initializer,
        },
        {
            TensorSlotName::BETA,
            beta_initializer,
        },
    };
  } else {
    return std::map<TensorSlotName, InitializerAttrs>{};
  }
}

} // namespace FlexFlow

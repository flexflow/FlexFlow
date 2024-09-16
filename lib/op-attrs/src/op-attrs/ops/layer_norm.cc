#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/dim_ordered/ff_ordered_of.h"
#include "op-attrs/dim_ordered/get_idxs.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/all_of.h"
#include "utils/containers/any_of.h"
#include "utils/containers/contains.h"
#include "utils/containers/extend.h"
#include "utils/containers/filter.h"

namespace FlexFlow {

std::vector<IncomingTensorRole>
    get_layer_norm_incoming_tensor_roles(LayerNormAttrs const &attrs) {
  std::vector<IncomingTensorRole> result = {IncomingTensorRole::INPUT};

  if (attrs.elementwise_affine) {
    extend(result,
           std::vector{IncomingTensorRole::WEIGHT, IncomingTensorRole::WEIGHT});
  }

  return result;
}

static std::optional<std::string>
    check_input_shape(LayerNormAttrs const &attrs,
                      TensorShape const &input_shape) {
  if (any_of(attrs.axes, [&](ff_dim_t axis) {
        return axis.value >= num_dims(input_shape);
      })) {
    return fmt::format(
        "LayerNorm axes {} out-of-bounds for input tensor shape {}",
        attrs.axes,
        input_shape);
  }

  return std::nullopt;
}

tl::expected<TensorShape, std::string>
    get_output_shape(LayerNormAttrs const &attrs,
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
    get_gamma_weights_shape(LayerNormAttrs const &attrs,
                            TensorShape const &input_shape) {
  {
    std::optional<std::string> maybe_err_msg =
        check_input_shape(attrs, input_shape);
    if (maybe_err_msg.has_value()) {
      return tl::unexpected(maybe_err_msg.value());
    }
  }

  if (!attrs.elementwise_affine) {
    return tl::unexpected(
        "No gamma weights exist for attrs.elementwise_affine = false");
  }

  std::vector<ff_dim_t> non_layer_norm_dim_idxs = filter(
      get_idxs(input_shape.dims.ff_ordered),
      [&](ff_dim_t const &dim_idx) { return !contains(attrs.axes, dim_idx); });
  std::vector<size_t> raw_weight_dims =
      transform(non_layer_norm_dim_idxs, [&](ff_dim_t const &dim_idx) {
        return dim_at_idx(input_shape, dim_idx);
      });

  return TensorShape{
      TensorDims{ff_ordered_of(raw_weight_dims)},
      DataType::FLOAT,
  };
}

tl::expected<TensorShape, std::string>
    get_beta_weights_shape(LayerNormAttrs const &attrs,
                           TensorShape const &input_shape) {
  if (!attrs.elementwise_affine) {
    return tl::unexpected(
        "No beta weights exist for attrs.elementwise_affine = false");
  }

  return get_gamma_weights_shape(attrs, input_shape);
}

static std::optional<std::string>
    check_input_shape(LayerNormAttrs const &attrs,
                      ParallelTensorShape const &input_shape) {
  {
    TensorShape reduced_shape = get_reduced_shape(input_shape);
    std::optional<std::string> maybe_err_msg =
        check_input_shape(attrs, reduced_shape);
    if (maybe_err_msg.has_value()) {
      return maybe_err_msg;
    }
  }

  if (get_sum_degree(input_shape) != 1) {
    return fmt::format("Expected sum degree 1, but receieved sum degree {}",
                       get_sum_degree(input_shape));
  }

  if (get_discard_copy_degree(input_shape) != 1) {
    return fmt::format(
        "Expected discard copy degree 1, but received discartd copy degree {}",
        get_discard_copy_degree(input_shape));
  }

  if (!all_of(attrs.axes, [&](ff_dim_t axis) {
        return shard_dim_at_idx(input_shape, axis).degree == 1;
      })) {
    return fmt::format("Expected parallel degree of all dimensions in "
                       "LayerNorm axes {} to be 1, but received input shape {}",
                       attrs.axes,
                       input_shape);
  }

  return std::nullopt;
}

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(LayerNormAttrs const &attrs,
                     ParallelTensorShape const &input_shape) {
  {
    std::optional<std::string> maybe_err_msg =
        check_input_shape(attrs, input_shape);
    if (maybe_err_msg.has_value()) {
      return tl::unexpected(maybe_err_msg.value());
    }
  }

  return input_shape;
}

tl::expected<ParallelTensorShape, std::string>
    get_gamma_weights_shape(LayerNormAttrs const &attrs,
                            ParallelTensorShape const &input_shape) {
  {
    std::optional<std::string> maybe_err_msg =
        check_input_shape(attrs, input_shape);
    if (maybe_err_msg.has_value()) {
      return tl::unexpected(maybe_err_msg.value());
    }
  }

  if (!attrs.elementwise_affine) {
    return tl::unexpected(
        "No gamma weights exist for attrs.elementwise_affine = false");
  }

  std::vector<ff_dim_t> non_layer_norm_dim_idxs = filter(
      get_idxs(input_shape.dims.shard_dims),
      [&](ff_dim_t const &dim_idx) { return !contains(attrs.axes, dim_idx); });
  std::vector<ShardParallelDim> raw_weight_shard_dims =
      transform(non_layer_norm_dim_idxs, [&](ff_dim_t const &dim_idx) {
        return shard_dim_at_idx(input_shape, dim_idx);
      });

  return ParallelTensorShape{
      ParallelTensorDims{
          ff_ordered_of(raw_weight_shard_dims),
          ReplicaParallelDimSet{
              SumDegree{1},
              DiscardCopyDegree{1},
          },
      },
      DataType::FLOAT,
  };
}

tl::expected<ParallelTensorShape, std::string>
    get_beta_weights_shape(LayerNormAttrs const &attrs,
                           ParallelTensorShape const &input_shape) {
  if (!attrs.elementwise_affine) {
    return tl::unexpected(
        "No beta weights exist for attrs.elementwise_affine = false");
  }

  return get_gamma_weights_shape(attrs, input_shape);
}

} // namespace FlexFlow

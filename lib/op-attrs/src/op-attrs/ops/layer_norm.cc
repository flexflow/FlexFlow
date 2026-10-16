#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/ff_ordered/ff_ordered_get_idxs.h"
#include "op-attrs/ff_ordered/ff_ordered_of.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/all_of.h"
#include "utils/containers/any_of.h"
#include "utils/containers/contains.h"
#include "utils/containers/extend.h"
#include "utils/containers/filter.h"
#include "utils/containers/vector_of.h"
#include "utils/expected.h"
#include "utils/fmt/set.h"

namespace FlexFlow {

std::map<TensorSlotName, IncomingTensorRole>
    get_layer_norm_incoming_tensor_roles(LayerNormAttrs const &attrs) {
  std::map<TensorSlotName, IncomingTensorRole> result = {
      {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
  };

  if (attrs.elementwise_affine) {
    result[TensorSlotName::GAMMA] = IncomingTensorRole::WEIGHT;
    result[TensorSlotName::BETA] = IncomingTensorRole::WEIGHT;
  }

  return result;
}

static void check_input_shape(LayerNormAttrs const &attrs,
                              TensorShape const &input_shape) {

  ASSERT(
      all_of(attrs.axes,
             [&](ff_dim_t axis) -> bool {
               return axis.value < get_num_dims(input_shape.dims);
             }),
      fmt::format("LayerNorm axes {} out-of-bounds for input tensor shape {}",
                  attrs.axes,
                  input_shape));
}

TensorShape layer_norm_get_output_shape(LayerNormAttrs const &attrs,
                                        TensorShape const &input_shape) {

  check_input_shape(attrs, input_shape);

  return input_shape;
}

TensorShape layer_norm_get_gamma_weights_shape(LayerNormAttrs const &attrs,
                                               TensorShape const &input_shape) {
  check_input_shape(attrs, input_shape);

  ASSERT(attrs.elementwise_affine,
         "No gamma weights exist for attrs.elementwise_affine = false");

  std::vector<ff_dim_t> non_layer_norm_dim_idxs = filter(
      vector_of(ff_ordered_get_idxs(input_shape.dims.ff_ordered)),
      [&](ff_dim_t const &dim_idx) { return !contains(attrs.axes, dim_idx); });
  std::vector<positive_int> raw_weight_dims =
      transform(non_layer_norm_dim_idxs, [&](ff_dim_t const &dim_idx) {
        return dim_at_idx(input_shape.dims,
                          relative_ff_dim_t_from_ff_dim_t(dim_idx));
      });

  return TensorShape{
      TensorDims{ff_ordered_of(raw_weight_dims)},
      DataType::FLOAT,
  };
}

TensorShape layer_norm_get_beta_weights_shape(LayerNormAttrs const &attrs,
                                              TensorShape const &input_shape) {

  ASSERT(attrs.elementwise_affine,
         "No beta weights exist for attrs.elementwise_affine = false");

  return layer_norm_get_gamma_weights_shape(attrs, input_shape);
}

std::map<TensorSlotName, TensorShape>
    layer_norm_get_weight_shapes(LayerNormAttrs const &attrs,
                                 TensorShape const &input_shape) {

  TensorShape gamma_shape =
      layer_norm_get_gamma_weights_shape(attrs, input_shape);
  TensorShape beta_shape =
      layer_norm_get_beta_weights_shape(attrs, input_shape);

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

static void check_input_parallel_shape(LayerNormAttrs const &attrs,
                                       ParallelTensorShape const &input_shape) {

  check_input_shape(attrs, get_reduced_shape(input_shape));

  ASSERT(get_sum_degree(input_shape) == 1,
         fmt::format("Expected sum degree 1, but received sum degree {}",
                     get_sum_degree(input_shape)));

  ASSERT(
      get_discard_copy_degree(input_shape) == 1,
      fmt::format(
          "Expected discard copy degree 1, but received discard copy degree {}",
          get_discard_copy_degree(input_shape)));

  ASSERT(all_of(attrs.axes,
                [&](ff_dim_t axis) -> bool {
                  return shard_dim_at_idx(input_shape,
                                          relative_ff_dim_t_from_ff_dim_t(axis))
                             .degree == 1;
                }),
         fmt::format("Expected parallel degree of all dimensions in "
                     "LayerNorm axes {} to be 1, but received input shape {}",
                     attrs.axes,
                     input_shape));
}

ParallelTensorShape layer_norm_get_output_parallel_shape(
    LayerNormAttrs const &attrs, ParallelTensorShape const &input_shape) {

  check_input_parallel_shape(attrs, input_shape);

  return input_shape;
}

ParallelTensorShape layer_norm_get_gamma_weights_parallel_shape(
    LayerNormAttrs const &attrs, ParallelTensorShape const &input_shape) {

  check_input_parallel_shape(attrs, input_shape);

  ASSERT(attrs.elementwise_affine,
         "No gamma weights exist for attrs.elementwise_affine = false");

  std::vector<ff_dim_t> non_layer_norm_dim_idxs = filter(
      vector_of(ff_ordered_get_idxs(input_shape.dims.shard_dims)),
      [&](ff_dim_t const &dim_idx) { return !contains(attrs.axes, dim_idx); });
  std::vector<ShardParallelDim> raw_weight_shard_dims =
      transform(non_layer_norm_dim_idxs, [&](ff_dim_t const &dim_idx) {
        return shard_dim_at_idx(input_shape,
                                relative_ff_dim_t_from_ff_dim_t(dim_idx));
      });

  return ParallelTensorShape{
      ParallelTensorDims{
          ff_ordered_of(raw_weight_shard_dims),
          ReplicaParallelDimSet{
              SumDegree{1_p},
              DiscardCopyDegree{1_p},
          },
      },
      DataType::FLOAT,
  };
}

ParallelTensorShape layer_norm_get_beta_weights_parallel_shape(
    LayerNormAttrs const &attrs, ParallelTensorShape const &input_shape) {

  ASSERT(attrs.elementwise_affine,
         "No beta weights exist for attrs.elementwise_affine = false");

  return layer_norm_get_gamma_weights_parallel_shape(attrs, input_shape);
}

std::map<TensorSlotName, ParallelTensorShape>
    layer_norm_get_weight_parallel_shapes(
        LayerNormAttrs const &attrs, ParallelTensorShape const &input_shape) {

  ParallelTensorShape gamma_shape =
      layer_norm_get_gamma_weights_parallel_shape(attrs, input_shape);
  ParallelTensorShape beta_shape =
      layer_norm_get_beta_weights_parallel_shape(attrs, input_shape);

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

std::map<TensorSlotName, InitializerAttrs>
    layer_norm_get_initializers(LayerNormAttrs const &attrs) {
  if (attrs.elementwise_affine) {
    InitializerAttrs gamma_initializer =
        InitializerAttrs{ConstantInitializerAttrs{DataTypeValue{float{1}}}};

    InitializerAttrs beta_initializer =
        InitializerAttrs{ConstantInitializerAttrs{DataTypeValue{float{0}}}};

    return {
        {TensorSlotName::GAMMA, gamma_initializer},
        {TensorSlotName::BETA, beta_initializer},
    };
  } else {
    return {};
  }
}

} // namespace FlexFlow

#include "op-attrs/ops/linear.h"
#include "op-attrs/ff_ordered/ff_ordered_slice.h"
#include "op-attrs/ff_ordered/ff_ordered_transform.h"
#include "op-attrs/initializers/kaiming_initializer_mode.h"
#include "op-attrs/num_ptensor_shard_dims_t.h"
#include "op-attrs/num_tensor_dims_t.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/relative_ff_dim_t.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/product.h"
#include "utils/containers/set_of.h"
#include "utils/expected.h"
#include "utils/fmt/optional.h"
#include "utils/integer_conversions.h"
#include "utils/orthotope/dim_projection.h"
#include "utils/orthotope/down_projection.h"
#include "utils/orthotope/eq_projection.h"
#include "utils/orthotope/minimal_dim_domain_mapping.h"
#include "utils/orthotope/up_projection.h"

namespace FlexFlow {

std::map<TensorSlotName, IncomingTensorRole>
    get_linear_incoming_tensor_roles(LinearAttrs const &attrs) {
  std::map<TensorSlotName, IncomingTensorRole> result = {
      {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
      {TensorSlotName::WEIGHT, IncomingTensorRole::WEIGHT},
  };

  if (attrs.use_bias) {
    result[TensorSlotName::BIAS] = IncomingTensorRole::WEIGHT;
  }

  return result;
}

TensorShape linear_get_projection_shape(LinearAttrs const &attrs,
                                        TensorShape const &input_shape) {
  positive_int in_channels =
      dim_at_idx(input_shape.dims, relative_ff_dim_t{-1});

  return TensorShape{
      TensorDims{
          FFOrdered<positive_int>{attrs.out_channels, in_channels},
      },
      input_shape.data_type,
  };
}

TensorShape linear_get_bias_shape(LinearAttrs const &attrs,
                                  TensorShape const &input_shape) {
  return TensorShape{
      TensorDims{
          FFOrdered<positive_int>{attrs.out_channels},
      },
      input_shape.data_type,
  };
}

TensorShape linear_get_output_shape(LinearAttrs const &attrs,
                                    TensorShape const &input_shape) {
  TensorShape output_shape = input_shape;
  output_shape.dims.ff_ordered.at(relative_ff_dim_t{-1}) = attrs.out_channels;

  return output_shape;
}

std::map<TensorSlotName, TensorShape>
    linear_get_weight_shapes(LinearAttrs const &attrs,
                             TensorShape const &input_shape) {

  std::map<TensorSlotName, TensorShape> weight_shapes = {
      {
          TensorSlotName::WEIGHT,
          linear_get_projection_shape(attrs, input_shape),
      },
  };

  if (attrs.use_bias) {
    weight_shapes.insert({
        TensorSlotName::BIAS,
        linear_get_bias_shape(attrs, input_shape),
    });
  }

  return weight_shapes;
}

//! [parallel shape inference composition example]
ParallelTensorShape
    linear_get_projection_parallel_shape(LinearAttrs const &attrs,
                                         ParallelTensorShape const &input) {
  TensorShape unpar =
      linear_get_projection_shape(attrs, get_reduced_shape(input));

  ParallelTensorDimDegrees projection_degrees =
      linear_get_projection_parallel_dim_degrees(attrs,
                                                 get_parallel_degrees(input));

  return lift_shape_to_parallel_with_degrees(unpar, projection_degrees);
}
//! [parallel shape inference composition example]

ParallelTensorShape
    linear_get_bias_parallel_shape(LinearAttrs const &attrs,
                                   ParallelTensorShape const &input) {
  TensorShape unpar = linear_get_bias_shape(attrs, get_reduced_shape(input));

  ParallelTensorDimDegrees bias_degrees =
      linear_get_bias_parallel_dim_degrees(attrs, get_parallel_degrees(input));

  return lift_shape_to_parallel_with_degrees(unpar, bias_degrees);
}

ParallelTensorShape
    linear_get_output_parallel_shape(LinearAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  TensorShape unpar = linear_get_output_shape(attrs, get_reduced_shape(input));

  ParallelTensorDimDegrees output_degrees =
      linear_get_output_parallel_dim_degrees(attrs,
                                             get_parallel_degrees(input));

  return lift_shape_to_parallel_with_degrees(unpar, output_degrees);
}

ParallelTensorDimDegrees linear_get_projection_parallel_dim_degrees(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input) {
  SumDegree sum_degree = SumDegree{1_p};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{
      input.sum_degree.value *
      product(ff_ordered_slice(
          input.shard_degrees, relative_ff_dim_t{0}, relative_ff_dim_t{-1}))};
  FFOrdered<positive_int> shard_degrees = FFOrdered<positive_int>{
      input.discard_copy_degree.value,
      input.shard_degrees.at(relative_ff_dim_t{-1}),
  };

  return ParallelTensorDimDegrees{
      /*sum_degree=*/sum_degree,
      /*discard_copy_degree=*/discard_copy_degree,
      /*shard_degrees=*/shard_degrees,
  };
}

ParallelTensorDimDegrees linear_get_bias_parallel_dim_degrees(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input) {

  SumDegree sum_degree = SumDegree{
      input.sum_degree.value * input.shard_degrees.at(relative_ff_dim_t{-1}),
  };
  DiscardCopyDegree discard_copy_degree =
      DiscardCopyDegree{product(ff_ordered_slice(
          input.shard_degrees, relative_ff_dim_t{0}, relative_ff_dim_t{-1}))};
  FFOrdered<positive_int> shard_degrees =
      FFOrdered<positive_int>{input.discard_copy_degree.value};

  return ParallelTensorDimDegrees{
      /*sum_degree=*/sum_degree,
      /*discard_copy_degree=*/discard_copy_degree,
      /*shard_degrees=*/shard_degrees,
  };
}

ParallelTensorDimDegrees linear_get_output_parallel_dim_degrees(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input) {
  SumDegree sum_degree = SumDegree{
      input.sum_degree.value * input.shard_degrees.at(relative_ff_dim_t{-1}),
  };

  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{1_p};
  FFOrdered<positive_int> shard_degrees = input.shard_degrees;
  shard_degrees.at(relative_ff_dim_t{-1}) = input.discard_copy_degree.value;

  return ParallelTensorDimDegrees{
      /*sum_degree=*/sum_degree,
      /*discard_copy_degree=*/discard_copy_degree,
      /*shard_degrees=*/shard_degrees,
  };
}

std::map<TensorSlotName, ParallelTensorShape>
    linear_get_weight_parallel_shapes(LinearAttrs const &attrs,
                                      ParallelTensorShape const &input_shape) {

  std::map<TensorSlotName, ParallelTensorShape> weight_shapes = {
      {
          TensorSlotName::WEIGHT,
          linear_get_projection_parallel_shape(attrs, input_shape),
      },
  };

  if (attrs.use_bias) {
    weight_shapes.insert({TensorSlotName::BIAS,
                          linear_get_bias_parallel_shape(attrs, input_shape)});
  }

  return weight_shapes;
}

/**
 * @brief Chosen to match pytorch implementation
 *
 * see
 * https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/nn/modules/linear.py#L114-L122
 */
std::map<TensorSlotName, InitializerAttrs> linear_get_initializers(
    LinearAttrs const &attrs,
    TensorShape const &input_shape,
    std::optional<InitializerAttrs> const &maybe_projection_initializer,
    std::optional<InitializerAttrs> const &maybe_bias_initializer) {

  ASSERT(attrs.use_bias || !maybe_bias_initializer.has_value(),
         fmt::format("Expected bias_initializer=std::nullopt since "
                     "use_bias=false, but received bias_initializer: {}",
                     maybe_bias_initializer.value()));

  TensorShape projection_shape =
      linear_get_projection_shape(attrs, input_shape);

  InitializerAttrs projection_default_initializer =
      InitializerAttrs{KaimingNormalAttrs{
          /*a=*/sqrtf(5.0),
          /*mode=*/KaimingInitializerMode::FAN_IN,
          /*nonlinearity=*/KaimingInitializerNonlinearity::LEAKY_RELU,
          /*seed=*/0,
      }};

  InitializerAttrs projection_initializer =
      maybe_projection_initializer.value_or(projection_default_initializer);

  positive_int fan_in = calculate_fan_for_mode(projection_shape.dims,
                                               KaimingInitializerMode::FAN_IN);

  float bound = 1 / sqrtf(static_cast<float>(fan_in.int_from_positive_int()));

  InitializerAttrs bias_default_initializer =
      InitializerAttrs{UniformInitializerAttrs{
          /*seed=*/0,
          /*min_val=*/-bound,
          /*max_val=*/bound,
      }};

  InitializerAttrs bias_initializer =
      maybe_bias_initializer.value_or(bias_default_initializer);

  if (attrs.use_bias) {
    return std::map<TensorSlotName, InitializerAttrs>{
        {TensorSlotName::WEIGHT, projection_initializer},
        {TensorSlotName::BIAS, bias_initializer},
    };
  } else {
    return std::map<TensorSlotName, InitializerAttrs>{
        {TensorSlotName::WEIGHT, projection_initializer},
    };
  }
}

OperatorTaskSpace linear_get_operator_task_space(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) {

  ParallelTensorDimDegrees output_degrees =
      linear_get_output_parallel_dim_degrees(attrs, input_degrees);

  return get_operator_task_space_matching_parallel_tensor_dim_degrees(
      output_degrees);
}

static ParallelTensorSpaceToParallelTensorSpaceMapping
    linear_get_input_to_output_mapping(
        LinearAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {

  num_tensor_dims_t input_num_dims =
      get_ptensor_dim_degrees_num_tensor_dims(input_degrees);

  DownProjection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>
      inp_to_out = make_empty_down_projection<parallel_tensor_dim_idx_t,
                                              parallel_tensor_dim_idx_t>();

  ff_dim_t input_channel_dim =
      ff_dim_t_from_relative_ff_dim_t(relative_ff_dim_t{-1}, input_num_dims);

  num_tensor_dims_t output_num_dims = input_num_dims;
  ff_dim_t output_channel_dim =
      ff_dim_t_from_relative_ff_dim_t(relative_ff_dim_t{-1}, output_num_dims);

  project_dims(inp_to_out,
               /*from=*/{sum_dim_idx(), shard_dim_idx(input_channel_dim)},
               /*onto=*/sum_dim_idx());
  project_dims(inp_to_out,
               /*from=*/{discard_copy_dim_idx()},
               /*onto=*/shard_dim_idx(output_channel_dim));

  for (ff_dim_t const &idx : slice(tensor_dims_range(input_num_dims), 0, -1)) {
    project_dims(inp_to_out,
                 /*from=*/{shard_dim_idx(idx)},
                 /*onto=*/shard_dim_idx(idx));
  }

  ParallelTensorDimDegrees output_degrees =
      linear_get_output_parallel_dim_degrees(attrs, input_degrees);

  return parallel_tensor_space_mapping_from_projection(
      DimProjection{inp_to_out}, input_degrees, output_degrees);
}

static ParallelTensorSpaceToParallelTensorSpaceMapping
    linear_get_input_to_projection_mapping(
        LinearAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {

  num_ptensor_shard_dims_t input_num_shard_dims =
      get_ptensor_dim_degrees_num_shard_dims(input_degrees);

  DownProjection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>
      inp_to_proj = make_empty_down_projection<parallel_tensor_dim_idx_t,
                                               parallel_tensor_dim_idx_t>();

  parallel_tensor_dim_idx_t input_channel_dim = parallel_tensor_dim_idx_t{
      ff_dim_t{
          nonnegative_int{
              input_num_shard_dims.value.unwrap_nonnegative() - 1,
          },
      },
  };

  {
    std::set<parallel_tensor_dim_idx_t> dims_from =
        set_of(dim_idxs_for_num_shard_dims(input_num_shard_dims));
    dims_from.insert(sum_dim_idx());
    dims_from.erase(input_channel_dim);
    dims_from.erase(discard_copy_dim_idx());

    project_dims(inp_to_proj,
                 /*from=*/dims_from,
                 /*onto=*/discard_copy_dim_idx());
  }

  parallel_tensor_dim_idx_t projection_in_channel_dim =
      parallel_tensor_dim_idx_t{ff_dim_t{0_n}};

  parallel_tensor_dim_idx_t projection_out_channel_dim =
      parallel_tensor_dim_idx_t{ff_dim_t{1_n}};

  project_dims(inp_to_proj,
               /*from=*/{discard_copy_dim_idx()},
               /*onto=*/projection_out_channel_dim);

  project_dims(inp_to_proj,
               /*from=*/{input_channel_dim},
               /*onto=*/projection_in_channel_dim);

  ParallelTensorDimDegrees projection_degrees =
      linear_get_projection_parallel_dim_degrees(attrs, input_degrees);

  return parallel_tensor_space_mapping_from_projection(
      DimProjection{inp_to_proj}, input_degrees, projection_degrees);
}

static ParallelTensorSpaceToParallelTensorSpaceMapping
    linear_get_input_to_bias_mapping(
        LinearAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {
  ASSERT(attrs.use_bias);

  num_ptensor_shard_dims_t input_num_shard_dims =
      get_ptensor_dim_degrees_num_shard_dims(input_degrees);

  ParallelTensorDimDegrees bias_degrees =
      linear_get_bias_parallel_dim_degrees(attrs, input_degrees);

  DownProjection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>
      inp_to_bias = make_empty_down_projection<parallel_tensor_dim_idx_t,
                                               parallel_tensor_dim_idx_t>();

  parallel_tensor_dim_idx_t input_channel_dim = parallel_tensor_dim_idx_t{
      ff_dim_t{
          nonnegative_int{
              input_num_shard_dims.value.unwrap_nonnegative() - 1,
          },
      },
  };

  {
    std::set<parallel_tensor_dim_idx_t> dims_from =
        set_of(dim_idxs_for_num_shard_dims(input_num_shard_dims));
    dims_from.erase(input_channel_dim);
    dims_from.erase(sum_dim_idx());

    project_dims(inp_to_bias,
                 /*from=*/dims_from,
                 /*onto=*/discard_copy_dim_idx());
  }

  parallel_tensor_dim_idx_t bias_out_channel_dim =
      parallel_tensor_dim_idx_t{ff_dim_t{0_n}};

  project_dims(inp_to_bias,
               /*from=*/
               {
                   sum_dim_idx(),
                   input_channel_dim,
               },
               /*onto=*/sum_dim_idx());

  DimDomain<parallel_tensor_dim_idx_t> l_domain =
      dim_domain_from_parallel_tensor_dim_degrees(input_degrees);
  DimDomain<parallel_tensor_dim_idx_t> r_domain =
      dim_domain_from_parallel_tensor_dim_degrees(bias_degrees);

  return parallel_tensor_space_mapping_from_projection(
      DimProjection{inp_to_bias}, input_degrees, bias_degrees);
}

OperatorSpaceToParallelTensorSpaceMapping
    linear_get_operator_to_projection_mapping(
        LinearAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {

  return operator_ptensor_space_mapping_from_composition(
      linear_get_operator_to_input_mapping(attrs, input_degrees),
      linear_get_input_to_projection_mapping(attrs, input_degrees));
}

OperatorSpaceToParallelTensorSpaceMapping linear_get_operator_to_input_mapping(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) {

  DimDomainMapping<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>
      inp_to_out =
          linear_get_input_to_output_mapping(attrs, input_degrees).raw_mapping;

  DimDomainMapping<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t>
      op_to_out = linear_get_operator_to_output_mapping(attrs, input_degrees)
                      .raw_mapping;

  DimDomainMapping<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t>
      op_to_inp = compose_dim_domain_mappings_through_minimal(
          op_to_out, invert_dim_domain_mapping(inp_to_out));

  return OperatorSpaceToParallelTensorSpaceMapping{
      op_to_inp,
  };
}

OperatorSpaceToParallelTensorSpaceMapping linear_get_operator_to_bias_mapping(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) {

  return operator_ptensor_space_mapping_from_composition(
      linear_get_operator_to_input_mapping(attrs, input_degrees),
      linear_get_input_to_bias_mapping(attrs, input_degrees));
}

OperatorSpaceToParallelTensorSpaceMapping linear_get_operator_to_output_mapping(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) {

  ParallelTensorDimDegrees output_degrees =
      linear_get_output_parallel_dim_degrees(attrs, input_degrees);

  return get_identity_mapping(
      linear_get_operator_task_space(attrs, input_degrees), output_degrees);
}

} // namespace FlexFlow

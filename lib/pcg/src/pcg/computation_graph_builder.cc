#include "pcg/computation_graph_builder.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/ff_ordered/ff_ordered_of.h"
#include "op-attrs/get_incoming_tensor_roles.h"
#include "op-attrs/get_op_type.h"
#include "op-attrs/ops/attention.h"
#include "op-attrs/ops/attention_attrs.dtg.h"
#include "op-attrs/ops/batch_norm.h"
#include "op-attrs/ops/batch_norm_attrs.dtg.h"
#include "op-attrs/ops/broadcast_attrs.dtg.h"
#include "op-attrs/ops/cast_attrs.dtg.h"
#include "op-attrs/ops/concat_attrs.dtg.h"
#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/ops/conv_2d_attrs.dtg.h"
#include "op-attrs/ops/dropout_attrs.dtg.h"
#include "op-attrs/ops/element_binary_attrs.dtg.h"
#include "op-attrs/ops/element_unary_attrs.dtg.h"
#include "op-attrs/ops/embedding.h"
#include "op-attrs/ops/embedding_attrs.dtg.h"
#include "op-attrs/ops/flat_attrs.dtg.h"
#include "op-attrs/ops/gather_attrs.dtg.h"
#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/ops/layer_norm_attrs.dtg.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/linear_attrs.dtg.h"
#include "op-attrs/ops/pool_2d.h"
#include "op-attrs/ops/pool_2d_attrs.dtg.h"
#include "op-attrs/ops/softmax_attrs.dtg.h"
#include "op-attrs/ops/upsample_attrs.dtg.h"
#include "op-attrs/ops/weight_attrs.dtg.h"
#include "op-attrs/relative_ff_dim_t.h"
#include "op-attrs/shape_inference.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "op-attrs/tensor_slot_name.h"
#include "pcg/computation_graph.h"
#include "utils/containers/any_of.h"
#include "utils/containers/are_all_distinct.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/enumerate_vector.h"
#include "utils/containers/get_only.h"
#include "utils/containers/repeat_element.h"
#include "utils/containers/require_only_key.h"
#include "utils/containers/slice.h"
#include "utils/containers/transform.h"
#include "utils/containers/transform_until.h"
#include "utils/containers/vector_of.h"
#include "utils/containers/without_nullopts.h"
#include "utils/containers/zip_values_strict_with.h"
#include "utils/containers/zip_with_strict.h"
#include "utils/exception.h"
#include "utils/expected.h"
#include "utils/fmt/set.h"
#include "utils/stack_vector/stack_vector_of.h"
#include <fmt/format.h>

namespace FlexFlow {

static std::string get_default_name(OperatorType op_type) {
  return get_operator_type_name(op_type);
}

static std::string get_default_name(ComputationGraphOpAttrs const &attrs) {
  return get_default_name(get_op_type(attrs));
}

ComputationGraphBuilder::ComputationGraphBuilder()
    : computation_graph(make_empty_computation_graph()) {}

TensorShape ComputationGraphBuilder::get_shape(tensor_guid_t const &t) const {
  return get_tensor_attrs(this->computation_graph, t).shape;
}

tensor_guid_t ComputationGraphBuilder::create_input(
    TensorShape const &shape,
    CreateGrad create_grad,
    std::optional<std::string> const &maybe_name) {

  LayerAttrs layer_attrs = LayerAttrs{
      ComputationGraphOpAttrs{InputAttrs{shape}},
      maybe_name,
  };

  return require_only_key(this->add_layer(/*layer=*/layer_attrs,
                                          /*inputs=*/{},
                                          /*weights=*/{},
                                          /*outputs=*/
                                          std::map<TensorSlotName, CreateGrad>{
                                              {
                                                  TensorSlotName::OUTPUT,
                                                  create_grad,
                                              },
                                          }),
                          TensorSlotName::OUTPUT);
}

tensor_guid_t ComputationGraphBuilder::create_weight(
    TensorShape const &shape,
    InitializerAttrs const &initializer,
    std::optional<std::string> const &maybe_name) {
  LayerAttrs layer_attrs = LayerAttrs{
      ComputationGraphOpAttrs{WeightAttrs{
          /*shape=*/shape,
          /*initializer=*/initializer,
      }},
      maybe_name,
  };

  return require_only_key(this->add_layer(layer_attrs, {}, {}),
                          TensorSlotName::OUTPUT);
}

static void
    check_incoming_tensor_roles(LayerAttrs const &layer,
                                std::set<TensorSlotName> const &input_slots,
                                std::set<TensorSlotName> const &weight_slots) {
  std::map<TensorSlotName, IncomingTensorRole> correct =
      restrict_keys(get_incoming_tensor_roles(layer.op_attrs),
                    set_union(input_slots, weight_slots));
  std::map<TensorSlotName, IncomingTensorRole> current =
      binary_merge_disjoint_maps(
          generate_map(
              input_slots,
              [](TensorSlotName) { return IncomingTensorRole::INPUT; }),
          generate_map(weight_slots, [](TensorSlotName) {
            return IncomingTensorRole::WEIGHT;
          }));

  ASSERT(correct == current,
         "check_incoming_tensor_roles found deviation in incoming tensors");
}

std::map<TensorSlotName, tensor_guid_t> ComputationGraphBuilder::add_layer(
    LayerAttrs const &layer,
    std::map<TensorSlotName, tensor_guid_t> const &inputs,
    std::map<TensorSlotName, InitializerAttrs> const &weight_initializers,
    std::optional<std::map<TensorSlotName, CreateGrad>> const &outputs) {
  ASSERT(are_disjoint(keys(inputs), keys(weight_initializers)));
  check_incoming_tensor_roles(layer, keys(inputs), keys(weight_initializers));

  std::map<TensorSlotName, TensorShape> input_shapes = map_values(
      inputs, [&](tensor_guid_t const &t) { return this->get_shape(t); });

  std::map<TensorSlotName, TensorShape> weight_shapes =
      get_weight_shapes(layer.op_attrs, input_shapes);

  std::map<TensorSlotName, tensor_guid_t> weights = zip_values_strict_with(
      weight_shapes,
      weight_initializers,
      [&](TensorShape const &shape, InitializerAttrs const &initializer) {
        return this->create_weight(shape, initializer);
      });

  LayerAddedResult added = ::FlexFlow::add_layer(
      this->computation_graph, layer, inputs, weights, outputs);
  return added.outputs;
}

tensor_guid_t ComputationGraphBuilder::as_type(tensor_guid_t const &x,
                                               DataType data_type,
                                               std::string const &name) {
  DataType x_datatype = this->get_shape(x).data_type;
  if (x_datatype < data_type) {
    return this->cast(x, data_type, name);
  } else if (x_datatype > data_type) {
    PANIC(fmt::format("Could not convert provided tensor data type {} to "
                      "desired data type {}",
                      x_datatype,
                      data_type));
  } else {
    return x;
  }
}

tensor_guid_t ComputationGraphBuilder::batch_matmul(
    tensor_guid_t const &lhs,
    tensor_guid_t const &rhs,
    std::optional<std::string> const &maybe_name) {
  BatchMatmulAttrs attrs = BatchMatmulAttrs{};

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::LHS_INPUT,
                                                  lhs,
                                              },
                                              {
                                                  TensorSlotName::RHS_INPUT,
                                                  rhs,
                                              },
                                          },
                                          {}),
                          TensorSlotName::OUTPUT);
}

tensor_guid_t ComputationGraphBuilder::broadcast(tensor_guid_t const &input,
                                                 TensorDims const &target_dims,
                                                 std::string const &name) {
  TensorShape input_shape = this->get_shape(input);
  if (input_shape.dims == target_dims) {
    return input;
  }

  ASSERT(tensor_dims_is_broadcastable_to(input_shape.dims, target_dims),
         "Cannot broadcast input tensor to target dims",
         input_shape.dims,
         target_dims);

  BroadcastAttrs attrs = BroadcastAttrs{target_dims};

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::INPUT,
                                                  input,
                                              },
                                          },
                                          {}),
                          TensorSlotName::OUTPUT);
}

tensor_guid_t ComputationGraphBuilder::cast(
    tensor_guid_t const &input,
    DataType dtype,
    std::optional<std::string> const &maybe_name) {

  CastAttrs attrs = CastAttrs{dtype};

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::INPUT,
                                                  input,
                                              },
                                          },
                                          {}),
                          TensorSlotName::OUTPUT);
}

tensor_guid_t ComputationGraphBuilder::element_unary(
    ElementUnaryOp op_type,
    tensor_guid_t const &x,
    std::optional<float> scalar,
    std::optional<std::string> const &maybe_name) {

  ElementUnaryAttrs attrs = ElementUnaryAttrs{op_type, scalar};

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  tensor_guid_t input =
      this->as_type(x, DataType::FLOAT, name + "input_pre_cast");

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::INPUT,
                                                  input,
                                              },
                                          },
                                          {}),
                          TensorSlotName::OUTPUT);
}

tensor_guid_t ComputationGraphBuilder::element_binary(
    ElementBinaryOp op_type,
    tensor_guid_t const &lhs,
    tensor_guid_t const &rhs,
    std::optional<std::string> const &maybe_name) {

  TensorDims compute_dims = this->get_broadcast_target_dims({lhs, rhs});
  DataType compute_type =
      std::max(this->get_shape(lhs).data_type, this->get_shape(rhs).data_type);

  ElementBinaryAttrs attrs =
      ElementBinaryAttrs{op_type, compute_type, false, false};

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  tensor_guid_t lhs_input = this->as_type(
      this->broadcast(
          lhs, compute_dims, fmt::format("{}_inputl_broadcast", name)),
      compute_type,
      name + "_inputl_cast");

  tensor_guid_t rhs_input = this->as_type(
      this->broadcast(
          rhs, compute_dims, fmt::format("{}_inputr_broadcast", name)),
      compute_type,
      name + "_inputr_cast");

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::LHS_INPUT,
                                                  lhs_input,
                                              },
                                              {
                                                  TensorSlotName::RHS_INPUT,
                                                  rhs_input,
                                              },
                                          },
                                          {}),
                          TensorSlotName::OUTPUT);
}

tensor_guid_t
    ComputationGraphBuilder::exp(tensor_guid_t const &input,
                                 std::optional<std::string> const &name) {
  return this->element_unary(ElementUnaryOp::EXP, input, std::nullopt, name);
}

tensor_guid_t
    ComputationGraphBuilder::add(tensor_guid_t const &lhs,
                                 tensor_guid_t const &rhs,
                                 std::optional<std::string> const &name) {
  return this->element_binary(ElementBinaryOp::ADD, lhs, rhs, name);
}

tensor_guid_t
    ComputationGraphBuilder::subtract(tensor_guid_t const &lhs,
                                      tensor_guid_t const &rhs,
                                      std::optional<std::string> const &name) {
  return this->element_binary(ElementBinaryOp::SUBTRACT, lhs, rhs, name);
}

tensor_guid_t
    ComputationGraphBuilder::multiply(tensor_guid_t const &lhs,
                                      tensor_guid_t const &rhs,
                                      std::optional<std::string> const &name) {
  return this->element_binary(ElementBinaryOp::MULTIPLY, lhs, rhs, name);
}

tensor_guid_t
    ComputationGraphBuilder::divide(tensor_guid_t const &lhs,
                                    tensor_guid_t const &rhs,
                                    std::optional<std::string> const &name) {
  return this->element_binary(ElementBinaryOp::DIVIDE, lhs, rhs, name);
}

tensor_guid_t
    ComputationGraphBuilder::max(tensor_guid_t const &lhs,
                                 tensor_guid_t const &rhs,
                                 std::optional<std::string> const &name) {
  return this->element_binary(ElementBinaryOp::MAX, lhs, rhs, name);
}

tensor_guid_t
    ComputationGraphBuilder::min(tensor_guid_t const &lhs,
                                 tensor_guid_t const &rhs,
                                 std::optional<std::string> const &name) {
  return this->element_binary(ElementBinaryOp::MIN, lhs, rhs, name);
}

tensor_guid_t
    ComputationGraphBuilder::rsqrt(tensor_guid_t const &input,
                                   std::optional<std::string> const &name) {
  return this->element_unary(ElementUnaryOp::RSQRT, input, std::nullopt, name);
}

tensor_guid_t
    ComputationGraphBuilder::pow(tensor_guid_t const &input,
                                 float exponent,
                                 std::optional<std::string> const &name) {
  return this->element_unary(ElementUnaryOp::POW, input, exponent, name);
}

tensor_guid_t ComputationGraphBuilder::scalar_multiply(
    tensor_guid_t const &input,
    float scalar,
    std::optional<std::string> const &name) {
  return this->element_unary(
      ElementUnaryOp::SCALAR_MULTIPLY, input, scalar, name);
}

tensor_guid_t ComputationGraphBuilder::scalar_add(
    tensor_guid_t const &input,
    float scalar,
    std::optional<std::string> const &name) {
  return this->element_unary(ElementUnaryOp::SCALAR_ADD, input, scalar, name);
}

tensor_guid_t ComputationGraphBuilder::scalar_sub(
    tensor_guid_t const &lhs,
    float rhs,
    std::optional<std::string> const &name) {
  return this->element_unary(ElementUnaryOp::SCALAR_SUB, lhs, rhs, name);
}

tensor_guid_t ComputationGraphBuilder::scalar_truediv(
    tensor_guid_t const &numerator,
    float denominator,
    std::optional<std::string> const &name) {
  return this->element_unary(
      ElementUnaryOp::SCALAR_TRUE_DIV, numerator, denominator, name);
}

tensor_guid_t
    ComputationGraphBuilder::sin(tensor_guid_t const &input,
                                 std::optional<std::string> const &name) {
  return this->element_unary(ElementUnaryOp::SIN, input, std::nullopt, name);
}

tensor_guid_t
    ComputationGraphBuilder::cos(tensor_guid_t const &input,
                                 std::optional<std::string> const &name) {
  return this->element_unary(ElementUnaryOp::COS, input, std::nullopt, name);
}

tensor_guid_t
    ComputationGraphBuilder::relu(tensor_guid_t const &input,
                                  std::optional<std::string> const &name) {
  return this->element_unary(ElementUnaryOp::RELU, input, std::nullopt, name);
}

tensor_guid_t
    ComputationGraphBuilder::identity(tensor_guid_t const &input,
                                      std::optional<std::string> const &name) {
  return this->element_unary(
      ElementUnaryOp::IDENTITY, input, std::nullopt, name);
}

tensor_guid_t
    ComputationGraphBuilder::gelu(tensor_guid_t const &input,
                                  std::optional<std::string> const &name) {
  return this->element_unary(ElementUnaryOp::GELU, input, std::nullopt, name);
}

tensor_guid_t
    ComputationGraphBuilder::sigmoid(tensor_guid_t const &input,
                                     std::optional<std::string> const &name) {
  return this->element_unary(
      ElementUnaryOp::SIGMOID, input, std::nullopt, name);
}

tensor_guid_t
    ComputationGraphBuilder::tanh(tensor_guid_t const &input,
                                  std::optional<std::string> const &name) {
  return this->element_unary(ElementUnaryOp::TANH, input, std::nullopt, name);
}

tensor_guid_t
    ComputationGraphBuilder::elu(tensor_guid_t const &input,
                                 std::optional<std::string> const &name) {
  return this->element_unary(ElementUnaryOp::ELU, input, std::nullopt, name);
}

tensor_guid_t
    ComputationGraphBuilder::silu(tensor_guid_t const &input,
                                  std::optional<std::string> const &name) {
  return this->element_unary(ElementUnaryOp::SILU, input, std::nullopt, name);
}

tensor_guid_t ComputationGraphBuilder::conv2d(
    tensor_guid_t const &x,
    positive_int outChannels,
    positive_int kernelH,
    positive_int kernelW,
    positive_int strideH,
    positive_int strideW,
    nonnegative_int paddingH,
    nonnegative_int paddingW,
    std::optional<Activation> const &activation,
    positive_int groups,
    bool use_bias,
    std::optional<InitializerAttrs> const &maybe_kernel_initializer,
    std::optional<InitializerAttrs> const &maybe_bias_initializer,
    std::optional<RegularizerAttrs> const &kernel_regularizer,
    std::optional<std::string> const &maybe_name) {
  Conv2DAttrs attrs = Conv2DAttrs{
      /*out_channels=*/outChannels,
      /*kernel_h=*/kernelH,
      /*kernel_w=*/kernelW,
      /*stride_h=*/strideH,
      /*stride_w=*/strideW,
      /*padding_h=*/paddingH,
      /*padding_w=*/paddingW,
      /*groups=*/groups,
      /*activation=*/activation,
      /*use_bias=*/use_bias,
  };

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  tensor_guid_t input =
      this->as_type(x, DataType::FLOAT, name + "input_pre_cast");

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  std::map<TensorSlotName, InitializerAttrs> initializers =
      conv2d_get_initializers(attrs,
                              this->get_shape(input),
                              maybe_kernel_initializer,
                              maybe_bias_initializer);

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::INPUT,
                                                  input,
                                              },
                                          },
                                          initializers),
                          TensorSlotName::OUTPUT);
}

tensor_guid_t ComputationGraphBuilder::upsample(
    tensor_guid_t const &input,
    int_ge_two scale_factor,
    UpsampleMode mode,
    std::optional<std::string> const &maybe_name) {

  UpsampleAttrs attrs = UpsampleAttrs{
      /*scale_factor=*/scale_factor,
      /*mode=*/mode,
  };

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::INPUT,
                                                  input,
                                              },
                                          },
                                          {}),
                          TensorSlotName::OUTPUT);
}

tensor_guid_t ComputationGraphBuilder::dropout(
    tensor_guid_t const &x,
    float rate,
    unsigned long long seed,
    std::optional<std::string> const &maybe_name) {
  DropoutAttrs attrs = DropoutAttrs{rate, seed};
  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};
  tensor_guid_t input =
      this->as_type(x, DataType::FLOAT, name + "input_pre_cast");

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::INPUT,
                                                  input,
                                              },
                                          },
                                          {}),
                          TensorSlotName::OUTPUT);
}

tensor_guid_t ComputationGraphBuilder::embedding(
    tensor_guid_t const &input,
    positive_int num_entries,
    positive_int outDim,
    AggregateOp aggr,
    DataType dtype,
    std::optional<InitializerAttrs> const &initializer,
    std::optional<std::string> const &maybe_name) {
  EmbeddingAttrs attrs = EmbeddingAttrs{
      /*num_entries=*/num_entries,
      /*out_channels=*/outDim,
      /*aggr=*/aggr,
      /*data_type=*/dtype,
  };
  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  TensorShape input_shape = this->get_shape(input);

  std::map<TensorSlotName, InitializerAttrs> initializers =
      get_initializers(attrs, initializer);

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::INPUT,
                                                  input,
                                              },
                                          },
                                          initializers),
                          TensorSlotName::OUTPUT);
}

tensor_guid_t ComputationGraphBuilder::gather(
    tensor_guid_t const &input,
    tensor_guid_t const &index,
    relative_ff_dim_t dim,
    std::optional<std::string> const &maybe_name) {
  if (this->get_shape(index).data_type != DataType::INT32 &&
      this->get_shape(index).data_type != DataType::INT64) {
    PANIC(fmt::format("Invalid data type for input tensor 2 for Gather: "
                      "{} (should be {} or {})",
                      this->get_shape(input).data_type,
                      DataType::INT32,
                      DataType::INT64));
  }

  GatherAttrs attrs = GatherAttrs{ff_dim_t_from_relative_ff_dim_t(
      dim, get_num_dims(this->get_shape(input).dims))};
  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::INPUT,
                                                  input,
                                              },
                                          },
                                          {}),
                          TensorSlotName::OUTPUT);
}

tensor_guid_t ComputationGraphBuilder::pool2d(
    tensor_guid_t const &x,
    positive_int kernelH,
    positive_int kernelW,
    positive_int strideH,
    positive_int strideW,
    nonnegative_int paddingH,
    nonnegative_int paddingW,
    PoolOp type,
    std::optional<Activation> const &activation,
    std::optional<std::string> const &maybe_name) {

  Pool2DAttrs attrs = Pool2DAttrs{
      /*kernel_h=*/kernelH,
      /*kernel_w=*/kernelW,
      /*stride_h=*/strideH,
      /*stride_w=*/strideW,
      /*padding_h=*/paddingH,
      /*padding_w=*/paddingW,
      /*pool_type=*/type,
      /*activation=*/activation,
  };

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  tensor_guid_t input =
      this->as_type(x, DataType::FLOAT, name + "input_pre_cast");

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::INPUT,
                                                  input,
                                              },
                                          },
                                          {}),
                          TensorSlotName::OUTPUT);
}

tensor_guid_t ComputationGraphBuilder::adaptive_pool2d(
    tensor_guid_t const &uncasted_input,
    positive_int output_h,
    positive_int output_w,
    PoolOp type,
    std::optional<Activation> const &activation,
    std::optional<std::string> const &maybe_name) {

  TensorDims input_dims = this->get_shape(uncasted_input).dims;

  Pool2DAttrs attrs = make_adaptive_pool2d_attrs(
      input_dims, output_h, output_w, type, activation);

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  tensor_guid_t casted_input =
      this->as_type(uncasted_input, DataType::FLOAT, name + "input_pre_cast");

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  TensorShape output_shape =
      get_output_shape(attrs, this->get_shape(casted_input));

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::INPUT,
                                                  casted_input,
                                              },
                                          },
                                          {}),
                          TensorSlotName::OUTPUT);
}

tensor_guid_t ComputationGraphBuilder::batch_norm(
    tensor_guid_t const &input,
    bool affine,
    std::optional<Activation> const &activation,
    float eps,
    std::optional<float> const &momentum,
    std::optional<std::string> const &maybe_name) {

  if (activation.has_value() && activation.value() != Activation::RELU) {
    PANIC(fmt::format(
        "batch_norm currently only supports (1) no activation function, or (2) "
        "relu activation function, but received {}. "
        "If you need support for additional activation functions, please "
        "create an issue.",
        activation));
  }

  BatchNormAttrs attrs = BatchNormAttrs{
      /*relu=*/activation,
      /*affine=*/affine,
      /*eps=*/eps,
      /*momentum=*/momentum,
  };

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  TensorShape input_shape = this->get_shape(input);

  std::map<TensorSlotName, InitializerAttrs> initializers =
      get_initializers(attrs);

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::INPUT,
                                                  input,
                                              },
                                          },
                                          initializers),
                          TensorSlotName::OUTPUT);
}

tensor_guid_t ComputationGraphBuilder::multihead_attention(
    tensor_guid_t const &query,
    tensor_guid_t const &key,
    tensor_guid_t const &value,
    positive_int embed_dim,
    positive_int num_heads,
    std::optional<positive_int> const &kdim,
    std::optional<positive_int> const &vdim,
    float dropout,
    bool bias,
    bool add_bias_kv,
    bool add_zero_attn,
    std::optional<InitializerAttrs> initializer,
    std::optional<std::string> const &maybe_name) {

  ASSERT(!add_bias_kv,
         "ComputationGraphBuilder::multihead_attention received currently "
         "unsupported argument add_bias_kv=true. "
         "If you need this functionality, please create an issue.");

  ASSERT(!add_zero_attn,
         "ComputationGraphBuilder::multihead_attention received currently "
         "unsupported argument add_zero_attn=true. "
         "If you need this functionality, please create an issue.");

  MultiHeadAttentionAttrs attrs = MultiHeadAttentionAttrs{
      /*embed_dim=*/embed_dim,
      /*num_heads=*/num_heads,
      /*kdim=*/kdim.value_or(embed_dim),
      /*vdim=*/vdim.value_or(embed_dim),
      /*dropout=*/dropout,
      /*bias=*/bias,
      /*add_bias_kv=*/add_bias_kv,
      /*add_zero_attn=*/add_zero_attn,
  };

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  std::map<TensorSlotName, InitializerAttrs> initializers =
      get_initializers(attrs,
                       this->get_shape(query),
                       this->get_shape(key),
                       this->get_shape(value),
                       initializer);

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::KEY,
                                                  query,
                                              },
                                              {
                                                  TensorSlotName::QUERY,
                                                  key,
                                              },
                                              {
                                                  TensorSlotName::VALUE,
                                                  value,
                                              },
                                          },
                                          initializers),
                          TensorSlotName::OUTPUT);
}

TensorDims ComputationGraphBuilder::get_broadcast_target_dims(
    std::vector<tensor_guid_t> const &inputs) {
  std::vector<TensorDims> inputs_dims = transform(
      inputs, [&](tensor_guid_t const &t) { return this->get_shape(t).dims; });

  return this->get_broadcast_target_dims(inputs_dims);
}

TensorDims ComputationGraphBuilder::get_broadcast_target_dims(
    std::vector<TensorDims> const &inputs_dims) {
  std::optional<TensorDims> maybe_result =
      ::FlexFlow::get_broadcast_target_dims(set_of(inputs_dims));

  if (maybe_result.has_value()) {
    return maybe_result.value();
  } else {
    PANIC(fmt::format(
        "ComputationGraphBuilder::get_broadcast_target_dims failed to find "
        "target tensor dims for input tensor dims {}",
        inputs_dims));
  }
}

tensor_guid_t ComputationGraphBuilder::dense(
    tensor_guid_t const &input,
    positive_int outDim,
    std::optional<Activation> activation,
    bool use_bias,
    DataType data_type,
    std::optional<InitializerAttrs> const &maybe_projection_initializer,
    std::optional<InitializerAttrs> const &maybe_bias_initializer,
    std::optional<std::string> const &maybe_name) {
  LinearAttrs attrs = LinearAttrs{
      /*out_channels=*/outDim,
      /*use_bias=*/use_bias,
      /*data_type=*/data_type,
      /*activation=*/activation,
      /*regularizer=*/std::nullopt,
  };

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  std::map<TensorSlotName, InitializerAttrs> initializers =
      get_initializers(attrs,
                       this->get_shape(input),
                       maybe_projection_initializer,
                       maybe_bias_initializer);

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::INPUT,
                                                  input,
                                              },
                                          },
                                          initializers),
                          TensorSlotName::OUTPUT);
}

tensor_guid_t ComputationGraphBuilder::concat(
    std::vector<tensor_guid_t> const &inputs,
    relative_ff_dim_t axis,
    std::optional<std::string> const &maybe_name) {

  std::vector<TensorSlotName> input_slot_names =
      get_variadic_inputs_slot_name_sequence();
  ASSERT(inputs.size() <= input_slot_names.size());

  ff_dim_t abs_axis = ff_dim_t_from_relative_ff_dim_t(
      axis, get_num_dims(this->get_shape(inputs.at(0)).dims));

  ConcatAttrs attrs = ConcatAttrs{
      /*axis=*/abs_axis,
      /*num_inputs=*/int_ge_two{inputs.size()},
  };

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  return require_only_key(
      this->add_layer(layer, map_from_pairs(zip(input_slot_names, inputs)), {}),
      TensorSlotName::OUTPUT);
}

tensor_guid_t ComputationGraphBuilder::flat(
    tensor_guid_t const &input,
    relative_ff_dim_t start_dim,
    std::optional<relative_ff_dim_t> const &end_dim,
    std::optional<std::string> const &maybe_name) {
  num_tensor_dims_t input_num_dims = get_num_dims(this->get_shape(input).dims);

  ff_dim_t abs_start_dim =
      ff_dim_t_from_relative_ff_dim_t(start_dim, input_num_dims);

  ff_dim_t abs_end_dim = ff_dim_t_from_relative_ff_dim_t(
      end_dim.value_or(
          relative_ff_dim_t{input_num_dims.int_from_num_tensor_dims() - 1}),
      input_num_dims);

  FlatAttrs attrs = FlatAttrs{
      /*start_dim=*/abs_start_dim,
      /*end_dim=*/abs_end_dim,
  };

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::INPUT,
                                                  input,
                                              },
                                          },
                                          {}),
                          TensorSlotName::OUTPUT);
}

tensor_guid_t ComputationGraphBuilder::layer_norm(
    tensor_guid_t const &input,
    std::set<relative_ff_dim_t> const &relative_axes,
    bool elementwise_affine,
    float eps,
    std::optional<std::string> const &maybe_name) {

  TensorShape input_shape = this->get_shape(input);

  auto resolve_dim_idx = [&](relative_ff_dim_t dim_idx) {
    return ff_dim_t_from_relative_ff_dim_t(dim_idx,
                                           get_num_dims(input_shape.dims));
  };

  std::set<ff_dim_t> axes = transform(relative_axes, resolve_dim_idx);

  if (any_of(axes, [&](ff_dim_t axis) {
        return axis.value >= get_num_dims(input_shape.dims);
      })) {
    PANIC(fmt::format(
        "ComputationGraphBuilder::layer_norm received axes {} with "
        "out-of-bound element (input tensor has num dimensions = {})",
        axes,
        get_num_dims(input_shape.dims)));
  }

  LayerNormAttrs attrs = LayerNormAttrs{
      /*axes=*/axes,
      /*elementwise_affine=*/elementwise_affine,
      /*eps=*/eps,
  };

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  std::map<TensorSlotName, InitializerAttrs> initializers =
      get_initializers(attrs);

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::INPUT,
                                                  input,
                                              },
                                          },
                                          initializers),
                          TensorSlotName::OUTPUT);
}

tensor_guid_t ComputationGraphBuilder::reshape(
    tensor_guid_t const &input,
    std::vector<positive_int> const &shape,
    std::optional<std::string> const &maybe_name) {
  TensorShape input_shape = this->get_shape(input);

  ReshapeAttrs attrs = ReshapeAttrs{
      /*shape=*/TensorShape{
          /*dims=*/TensorDims{ff_ordered_of(shape)},
          /*data_type=*/input_shape.data_type,
      },
  };

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::INPUT,
                                                  input,
                                              },
                                          },
                                          {}),
                          TensorSlotName::OUTPUT);
}

tensor_guid_t ComputationGraphBuilder::softmax(
    tensor_guid_t const &input,
    std::optional<relative_ff_dim_t> maybe_dim,
    std::optional<std::string> const &maybe_name) {

  TensorShape input_shape = this->get_shape(input);

  relative_ff_dim_t dim = maybe_dim.value_or(relative_ff_dim_t{
      get_num_dims(input_shape.dims).int_from_num_tensor_dims() - 1});

  SoftmaxAttrs attrs = SoftmaxAttrs{
      ff_dim_t_from_relative_ff_dim_t(dim, get_num_dims(input_shape.dims))};

  ASSERT(attrs.dim.value < get_num_dims(input_shape.dims),
         "ComputationGraphBuilder::softmax received out_of_bounds dim",
         attrs.dim,
         input_shape);

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::INPUT,
                                                  input,
                                              },
                                          },
                                          {}),
                          TensorSlotName::OUTPUT);
}

std::vector<tensor_guid_t> ComputationGraphBuilder::split(
    tensor_guid_t const &input,
    std::vector<positive_int> const &split,
    relative_ff_dim_t axis,
    std::optional<std::string> const &maybe_name) {
  ASSERT(split.size() > 0);
  ASSERT(split.size() <= get_variadic_inputs_slot_name_sequence().size());

  TensorShape input_shape = this->get_shape(input);

  SplitAttrs attrs = SplitAttrs{
      /*splits=*/stack_vector_of<MAX_NUM_OUTPUTS>(split),
      /*axis=*/
      ff_dim_t_from_relative_ff_dim_t(axis, get_num_dims(input_shape.dims)),
  };

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  std::map<TensorSlotName, tensor_guid_t> outputs =
      this->add_layer(layer,
                      {
                          {
                              TensorSlotName::INPUT,
                              input,
                          },
                      },
                      {});

  std::vector<TensorSlotName> expected_output_slot_names =
      slice(get_variadic_outputs_slot_name_sequence(), 0, split.size());

  return transform(
      expected_output_slot_names,
      [&](TensorSlotName s) -> tensor_guid_t { return outputs.at(s); });
}

tensor_guid_t ComputationGraphBuilder::transpose(
    tensor_guid_t const &input,
    std::vector<nonnegative_int> const &perm,
    std::optional<std::string> const &maybe_name) {
  TensorShape input_shape = this->get_shape(input);

  ASSERT(are_all_distinct(perm));
  ASSERT(get_num_dims(input_shape.dims) == perm.size());

  TransposeAttrs attrs = TransposeAttrs{
      /*permutation=*/
      TensorDimPermutation{
          bidict_from_keys_and_values(
              ff_dim_range(num_elements(perm)),
              transform(perm, [](nonnegative_int n) { return ff_dim_t{n}; })),
      },
  };

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::INPUT,
                                                  input,
                                              },
                                          },
                                          {}),
                          TensorSlotName::OUTPUT);
}

} // namespace FlexFlow

#include "pcg/computation_graph_builder.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/get_incoming_tensor_roles.h"
#include "op-attrs/get_op_type.h"
#include "op-attrs/get_output_shapes.h"
#include "op-attrs/ops/attention.h"
#include "op-attrs/ops/batch_norm.h"
#include "op-attrs/ops/broadcast.h"
#include "op-attrs/ops/concat.h"
#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/ops/dropout.h"
#include "op-attrs/ops/element_binary.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/embedding.h"
#include "op-attrs/ops/flat.h"
#include "op-attrs/ops/gather.h"
#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/pool_2d.h"
#include "op-attrs/ops/softmax.h"
#include "op-attrs/ops/weight_attrs.dtg.h"
#include "op-attrs/tensor_dims.h"
#include "pcg/computation_graph.h"
#include "utils/containers/any_of.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/enumerate_vector.h"
#include "utils/containers/get_only.h"
#include "utils/containers/transform.h"
#include "utils/expected.h"
#include <fmt/format.h>

namespace FlexFlow {

static TensorAttrs make_weight_attrs(
    TensorShape const &shape,
    std::optional<InitializerAttrs> const &initializer_attrs) {
  return TensorAttrs{shape, initializer_attrs, std::nullopt, CreateGrad::YES};
}

static TensorAttrs make_output_attrs(TensorShape const &shape) {
  return TensorAttrs{shape, std::nullopt, std::nullopt, CreateGrad::YES};
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
  TensorAttrs tensor_attrs =
      TensorAttrs{shape, std::nullopt, std::nullopt, create_grad};
  LayerAttrs layer_attrs = LayerAttrs{
      ComputationGraphOpAttrs{InputAttrs{}},
      maybe_name,
  };

  return get_only(this->add_layer(layer_attrs, {}, {}, {tensor_attrs}));
}

tensor_guid_t ComputationGraphBuilder::create_weight(
    TensorAttrs const &tensor_attrs,
    std::optional<std::string> const &maybe_name) {
  LayerAttrs layer_attrs = LayerAttrs{
      ComputationGraphOpAttrs{InputAttrs{}},
      maybe_name,
  };

  return get_only(this->add_layer(layer_attrs,
                                  std::vector<tensor_guid_t>{},
                                  std::vector<tensor_guid_t>{},
                                  {tensor_attrs}));
}

tensor_guid_t ComputationGraphBuilder::create_weight(
    TensorShape const &shape,
    CreateGrad create_grad,
    std::optional<InitializerAttrs> const &initializer,
    std::optional<ParamSync> param_sync,
    std::optional<std::string> const &maybe_name) {
  TensorAttrs tensor_attrs =
      TensorAttrs{shape, initializer, param_sync, create_grad};

  return this->create_weight(tensor_attrs, maybe_name);
}

static void check_incoming_tensor_roles(LayerAttrs const &layer,
                                        int num_inputs,
                                        int num_weights) {
  std::vector<IncomingTensorRole> correct =
      get_incoming_tensor_roles(layer.attrs, num_inputs + num_weights);
  std::vector<IncomingTensorRole> current = concat_vectors(
      std::vector<IncomingTensorRole>(num_inputs, IncomingTensorRole::INPUT),
      std::vector<IncomingTensorRole>(num_weights, IncomingTensorRole::WEIGHT));

  if (correct != current) {
    throw mk_runtime_error(
        fmt::format("check_incoming_tensor_roles found deviation in incoming "
                    "tensors: expected {}, received {}",
                    correct,
                    current));
  }
}

std::vector<tensor_guid_t> ComputationGraphBuilder::add_layer(
    LayerAttrs const &layer,
    std::vector<tensor_guid_t> const &inputs,
    std::vector<tensor_guid_t> const &weights,
    std::vector<TensorAttrs> const &outputs) {
  check_incoming_tensor_roles(layer, inputs.size(), weights.size());

  LayerAddedResult added = ::FlexFlow::add_layer(
      this->computation_graph, layer, concat_vectors(inputs, weights), outputs);
  return added.outputs;
}

tensor_guid_t ComputationGraphBuilder::as_type(tensor_guid_t const &x,
                                               DataType data_type,
                                               std::string const &name) {
  DataType x_datatype = this->get_shape(x).data_type;
  if (x_datatype < data_type) {
    return this->cast(x, data_type, name);
  } else if (x_datatype > data_type) {
    throw mk_runtime_error(
        fmt::format("Could not convert provided tensor data type {} to "
                    "desired data type {}",
                    x_datatype,
                    data_type));
  } else {
    return x;
  }
}

tensor_guid_t ComputationGraphBuilder::broadcast(tensor_guid_t const &input,
                                                 TensorDims const &target_dims,
                                                 std::string const &name) {
  TensorShape input_shape = this->get_shape(input);
  if (input_shape.dims == target_dims) {
    return input;
  }

  if (!tensor_dims_is_broadcastable_to(input_shape.dims, target_dims)) {
    throw mk_runtime_error(fmt::format(
        "Cannot broadcast input tensor of dims {} to target dims {}",
        input_shape.dims,
        target_dims));
  }

  BroadcastAttrs attrs = BroadcastAttrs{target_dims};

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};
  TensorShape output_shape =
      throw_if_unexpected(get_output_shape(attrs, input_shape));

  return get_only(
      this->add_layer(layer, {input}, {}, {make_output_attrs(output_shape)}));
}

tensor_guid_t
    ComputationGraphBuilder::cast(tensor_guid_t const &input,
                                  DataType dtype,
                                  std::optional<std::string> const &name) {
  NOT_IMPLEMENTED()
}

static std::string get_default_name(OperatorType op_type) {
  return get_operator_type_name(op_type);
}

static std::string get_default_name(ComputationGraphOpAttrs const &attrs) {
  return get_default_name(get_op_type(attrs));
}

tensor_guid_t ComputationGraphBuilder::element_unary(
    OperatorType op_type,
    tensor_guid_t const &x,
    std::optional<float> scalar,
    std::optional<std::string> const &maybe_name) {

  ElementUnaryAttrs attrs = ElementUnaryAttrs{op_type, scalar};

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  tensor_guid_t input =
      this->as_type(x, DataType::FLOAT, name + "input_pre_cast");

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  TensorShape output_shape =
      throw_if_unexpected(get_output_shape(attrs, this->get_shape(input)));

  return get_only(
      this->add_layer(layer, {input}, {}, {make_output_attrs(output_shape)}));
}

tensor_guid_t ComputationGraphBuilder::element_binary(
    OperatorType op_type,
    tensor_guid_t const &lhs,
    tensor_guid_t const &rhs,
    std::optional<std::string> const &maybe_name) {
  std::string name = maybe_name.value_or(get_default_name(op_type));

  TensorDims compute_dims = this->get_broadcast_target_dims({lhs, rhs});
  DataType compute_type =
      std::max(this->get_shape(lhs).data_type, this->get_shape(rhs).data_type);

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

  ElementBinaryAttrs attrs =
      ElementBinaryAttrs{op_type, compute_type, false, false};

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  TensorShape output_shape = throw_if_unexpected(get_output_shape(
      attrs, this->get_shape(lhs_input), this->get_shape(rhs_input)));

  return get_only(this->add_layer(
      layer, {lhs_input, rhs_input}, {}, {make_output_attrs(output_shape)}));
}

tensor_guid_t
    ComputationGraphBuilder::exp(tensor_guid_t const &input,
                                 std::optional<std::string> const &name) {
  return this->element_unary(OperatorType::EXP, input, std::nullopt, name);
}

tensor_guid_t
    ComputationGraphBuilder::add(tensor_guid_t const &lhs,
                                 tensor_guid_t const &rhs,
                                 std::optional<std::string> const &name) {
  return this->element_binary(OperatorType::EW_ADD, lhs, rhs, name);
}

tensor_guid_t
    ComputationGraphBuilder::subtract(tensor_guid_t const &lhs,
                                      tensor_guid_t const &rhs,
                                      std::optional<std::string> const &name) {
  return this->element_binary(OperatorType::EW_SUB, lhs, rhs, name);
}

tensor_guid_t
    ComputationGraphBuilder::multiply(tensor_guid_t const &lhs,
                                      tensor_guid_t const &rhs,
                                      std::optional<std::string> const &name) {
  return this->element_binary(OperatorType::EW_MUL, lhs, rhs, name);
}

tensor_guid_t
    ComputationGraphBuilder::divide(tensor_guid_t const &lhs,
                                    tensor_guid_t const &rhs,
                                    std::optional<std::string> const &name) {
  return this->element_binary(OperatorType::EW_DIV, lhs, rhs, name);
}

tensor_guid_t
    ComputationGraphBuilder::max(tensor_guid_t const &lhs,
                                 tensor_guid_t const &rhs,
                                 std::optional<std::string> const &name) {
  return this->element_binary(OperatorType::EW_MAX, lhs, rhs, name);
}

tensor_guid_t
    ComputationGraphBuilder::min(tensor_guid_t const &lhs,
                                 tensor_guid_t const &rhs,
                                 std::optional<std::string> const &name) {
  return this->element_binary(OperatorType::EW_MIN, lhs, rhs, name);
}

tensor_guid_t
    ComputationGraphBuilder::rsqrt(tensor_guid_t const &input,
                                   std::optional<std::string> const &name) {
  return this->element_unary(OperatorType::RSQRT, input, std::nullopt, name);
}

tensor_guid_t
    ComputationGraphBuilder::pow(tensor_guid_t const &input,
                                 float exponent,
                                 std::optional<std::string> const &name) {
  return this->element_unary(OperatorType::POW, input, exponent, name);
}

tensor_guid_t ComputationGraphBuilder::scalar_multiply(
    tensor_guid_t const &input,
    float scalar,
    std::optional<std::string> const &name) {
  return this->element_unary(
      OperatorType::SCALAR_MULTIPLY, input, scalar, name);
}

tensor_guid_t ComputationGraphBuilder::scalar_add(
    tensor_guid_t const &input,
    float scalar,
    std::optional<std::string> const &name) {
  return this->element_unary(OperatorType::SCALAR_ADD, input, scalar, name);
}

tensor_guid_t ComputationGraphBuilder::scalar_sub(
    tensor_guid_t const &lhs,
    float rhs,
    std::optional<std::string> const &name) {
  return this->element_unary(OperatorType::SCALAR_SUB, lhs, rhs, name);
}

tensor_guid_t ComputationGraphBuilder::scalar_truediv(
    tensor_guid_t const &numerator,
    float denominator,
    std::optional<std::string> const &name) {
  return this->element_unary(
      OperatorType::SCALAR_TRUE_DIV, numerator, denominator, name);
}

tensor_guid_t
    ComputationGraphBuilder::sin(tensor_guid_t const &input,
                                 std::optional<std::string> const &name) {
  return this->element_unary(OperatorType::SIN, input, std::nullopt, name);
}

tensor_guid_t
    ComputationGraphBuilder::cos(tensor_guid_t const &input,
                                 std::optional<std::string> const &name) {
  return this->element_unary(OperatorType::COS, input, std::nullopt, name);
}

tensor_guid_t
    ComputationGraphBuilder::relu(tensor_guid_t const &input,
                                  std::optional<std::string> const &name) {
  return this->element_unary(OperatorType::RELU, input, std::nullopt, name);
}

tensor_guid_t
    ComputationGraphBuilder::identity(tensor_guid_t const &input,
                                      std::optional<std::string> const &name) {
  return this->element_unary(OperatorType::IDENTITY, input, std::nullopt, name);
}

tensor_guid_t
    ComputationGraphBuilder::gelu(tensor_guid_t const &input,
                                  std::optional<std::string> const &name) {
  return this->element_unary(OperatorType::GELU, input, std::nullopt, name);
}

tensor_guid_t
    ComputationGraphBuilder::sigmoid(tensor_guid_t const &input,
                                     std::optional<std::string> const &name) {
  return this->element_unary(OperatorType::SIGMOID, input, std::nullopt, name);
}

tensor_guid_t
    ComputationGraphBuilder::tanh(tensor_guid_t const &input,
                                  std::optional<std::string> const &name) {
  return this->element_unary(OperatorType::TANH, input, std::nullopt, name);
}

tensor_guid_t
    ComputationGraphBuilder::elu(tensor_guid_t const &input,
                                 std::optional<std::string> const &name) {
  return this->element_unary(OperatorType::ELU, input, std::nullopt, name);
}

tensor_guid_t ComputationGraphBuilder::conv2d(
    tensor_guid_t const &x,
    int outChannels,
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int paddingH,
    int paddingW,
    std::optional<Activation> const &activation,
    int groups,
    bool use_bias,
    std::optional<InitializerAttrs> const &kernel_initializer,
    std::optional<InitializerAttrs> const &bias_initializer,
    std::optional<RegularizerAttrs> const &kernel_regularizer,
    std::optional<std::string> const &maybe_name) {
  Conv2DAttrs attrs = Conv2DAttrs{outChannels,
                                  kernelH,
                                  kernelW,
                                  strideH,
                                  strideW,
                                  paddingH,
                                  paddingW,
                                  groups,
                                  activation,
                                  use_bias};

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  tensor_guid_t input =
      this->as_type(x, DataType::FLOAT, name + "input_pre_cast");

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  TensorShape input_shape = this->get_shape(input);
  TensorShape output_shape = get_output_shape(attrs, input_shape);

  std::vector<TensorAttrs> weights;

  weights.push_back(make_weight_attrs(get_kernel_shape(attrs, input_shape),
                                      kernel_initializer));

  if (use_bias) {
    weights.push_back(make_weight_attrs(get_bias_shape(attrs, input_shape),
                                        bias_initializer));
  }

  return get_only(this->add_layer(
      layer,
      {input},
      transform(weights,
                [&](TensorAttrs const &a) { return this->create_weight(a); }),
      {make_output_attrs(output_shape)}));
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

  TensorShape output_shape = get_output_shape(attrs, this->get_shape(input));

  return get_only(
      this->add_layer(layer, {input}, {}, {make_output_attrs(output_shape)}));
}

tensor_guid_t ComputationGraphBuilder::embedding(
    tensor_guid_t const &x,
    int num_entries,
    int outDim,
    AggregateOp aggr,
    DataType dtype,
    std::optional<InitializerAttrs> const &kernel_initializer,
    std::optional<std::string> const &maybe_name) {
  EmbeddingAttrs attrs = EmbeddingAttrs{num_entries, outDim, aggr, dtype};
  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};
  tensor_guid_t input =
      this->as_type(x, DataType::FLOAT, name + "input_pre_cast");

  TensorShape input_shape = this->get_shape(input);

  TensorAttrs weight_attrs = make_weight_attrs(
      throw_if_unexpected(get_weights_shape(attrs, input_shape)),
      kernel_initializer);

  TensorShape output_shape =
      throw_if_unexpected(get_output_shape(attrs, this->get_shape(input)));

  return get_only(this->add_layer(layer,
                                  {input},
                                  {this->create_weight(weight_attrs)},
                                  {make_output_attrs(output_shape)}));
}

tensor_guid_t ComputationGraphBuilder::gather(
    tensor_guid_t const &input,
    tensor_guid_t const &index,
    ff_dim_t dim,
    std::optional<std::string> const &maybe_name) {
  GatherAttrs attrs = GatherAttrs{dim};
  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};
  if (this->get_shape(index).data_type != DataType::INT32 &&
      this->get_shape(index).data_type != DataType::INT64) {
    throw mk_runtime_error("Invalid data type for input tensor 2 for Gather: "
                           "{} (should be {} or {})",
                           this->get_shape(input).data_type,
                           DataType::INT32,
                           DataType::INT64);
  }
  TensorShape output_shape =
      get_output_shape(attrs, this->get_shape(input), this->get_shape(index));

  return get_only(
      this->add_layer(layer, {input}, {}, {make_output_attrs(output_shape)}));
}
tensor_guid_t ComputationGraphBuilder::pool2d(
    tensor_guid_t const &x,
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int paddingH,
    int paddingW,
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

  TensorShape output_shape =
      throw_if_unexpected(get_output_shape(attrs, this->get_shape(input)));

  return get_only(
    this->add_layer(layer, {input}, {}, {make_output_attrs(output_shape)})
  );
}

tensor_guid_t
    ComputationGraphBuilder::adaptive_pool2d(tensor_guid_t const &uncasted_input,
                    int output_h,
                    int output_w,
                    PoolOp type,
                    std::optional<Activation> const &activation,
                    std::optional<std::string> const &maybe_name) {

  TensorDims input_dims = this->get_shape(uncasted_input).dims;

  Pool2DAttrs attrs = throw_if_unexpected(make_adaptive_pool2d_attrs(input_dims,
                                                                     output_h,
                                                                     output_w,
                                                                     type,
                                                                     activation));

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  tensor_guid_t casted_input =
      this->as_type(uncasted_input, DataType::FLOAT, name + "input_pre_cast");

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  TensorShape output_shape =
      throw_if_unexpected(get_output_shape(attrs, this->get_shape(casted_input)));

  return get_only(
    this->add_layer(layer, {casted_input}, {}, {make_output_attrs(output_shape)})
  );
}

tensor_guid_t ComputationGraphBuilder::batch_norm(
    tensor_guid_t const &input,
    bool affine,
    std::optional<Activation> const &activation,
    float eps,
    std::optional<float> const &momentum,
    std::optional<std::string> const &maybe_name) {

  if (activation.has_value() && activation.value() != Activation::RELU) {
    throw mk_runtime_error(fmt::format("batch_norm currently only supports (1) no activation function, or (2) relu activation function, but received {}. "
                                       "If you need support for additional activation functions, please create an issue.", activation));
  }

  BatchNormAttrs attrs = BatchNormAttrs{
    /*relu=*/activation.has_value(),
    /*affine=*/affine,
    /*eps=*/eps,
    /*momentum=*/momentum,
  };

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  TensorShape input_shape = this->get_shape(input);
  TensorShape output_shape = throw_if_unexpected(get_output_shape(attrs, input_shape));

  std::vector<TensorAttrs> weights;

  if (affine) {
    // initializers chosen to match those of
    // https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html

    TensorShape gamma_shape =
        throw_if_unexpected(get_gamma_weights_shape(attrs, input_shape));
    InitializerAttrs gamma_initializer =
        InitializerAttrs{ConstantInitializerAttrs{DataTypeValue{float{1}}}};
    weights.push_back(make_weight_attrs(gamma_shape, gamma_initializer));

    TensorShape beta_shape =
        throw_if_unexpected(get_beta_weights_shape(attrs, input_shape));
    InitializerAttrs beta_initializer =
        InitializerAttrs{ConstantInitializerAttrs{DataTypeValue{float{0}}}};
    weights.push_back(make_weight_attrs(beta_shape, beta_initializer));
  }

  return get_only(
      this->add_layer(layer, 
                      {input}, 
                      transform(weights, 
                                [&](TensorAttrs const &a) { return this->create_weight(a); }),
                      {make_output_attrs(output_shape)}));
}

tensor_guid_t ComputationGraphBuilder::multihead_attention(
    tensor_guid_t const &query,
    tensor_guid_t const &key,
    tensor_guid_t const &value,
    int embed_dim,
    int num_heads,
    int kdim,
    int vdim,
    float dropout,
    bool bias,
    bool add_bias_kv,
    bool add_zero_attn,
    std::optional<InitializerAttrs> initializer,
    std::optional<std::string> const &maybe_name) {

  if (add_bias_kv) {
    throw mk_runtime_error(
        "ComputationGraphBuilder::multihead_attention received currently "
        "unsupported argument add_bias_kv=true. "
        "If you need this functionality, please create an issue.");
  }

  if (add_zero_attn) {
    throw mk_runtime_error(
        "ComputationGraphBuilder::multihead_attention received currently "
        "unsupported argument add_zero_attn=true. "
        "If you need this functionality, please create an issue.");
  }

  MultiHeadAttentionAttrs attrs = MultiHeadAttentionAttrs{embed_dim,
                                                          num_heads,
                                                          kdim,
                                                          vdim,
                                                          dropout,
                                                          bias,
                                                          add_bias_kv,
                                                          add_zero_attn};

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  TensorShape query_shape = this->get_shape(query);
  TensorShape key_shape = this->get_shape(key);
  TensorShape value_shape = this->get_shape(value);

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};
  TensorShape output_shape = throw_if_unexpected(
      get_output_shape(attrs, query_shape, key_shape, value_shape));

  std::vector<TensorAttrs> weights;

  TensorShape weights_shape = throw_if_unexpected(
      get_weights_shape(attrs, query_shape, key_shape, value_shape));
  weights.push_back(make_weight_attrs(weights_shape, initializer));

  if (bias) {
    TensorShape input_bias_shape = throw_if_unexpected(
        get_input_bias_shape(attrs, query_shape, key_shape, value_shape));
    // initializer chosen based on
    // https://github.com/pytorch/pytorch/blob/31c4e0d37d8efc37a0697159e5b9121ec34d5141/torch/nn/modules/activation.py#L1120-L1121
    InitializerAttrs input_bias_initializer =
        InitializerAttrs{ConstantInitializerAttrs{DataTypeValue{float{0}}}};

    weights.push_back(
        make_weight_attrs(input_bias_shape, input_bias_initializer));

    TensorShape output_bias_shape = throw_if_unexpected(
        get_output_bias_shape(attrs, query_shape, key_shape, value_shape));
    // initializer chosen based on
    // https://github.com/pytorch/pytorch/blob/31c4e0d37d8efc37a0697159e5b9121ec34d5141/torch/nn/modules/activation.py#L1120-L1121
    InitializerAttrs output_bias_initializer =
        InitializerAttrs{ConstantInitializerAttrs{DataTypeValue{float{0}}}};

    weights.push_back(
        make_weight_attrs(output_bias_shape, output_bias_initializer));
  }

  return get_only(this->add_layer(
      layer,
      {query, key, value},
      transform(weights,
                [&](TensorAttrs const &a) { return this->create_weight(a); }),
      {make_output_attrs(output_shape)}));
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
      ::FlexFlow::get_broadcast_target_dims(unordered_set_of(inputs_dims));

  if (maybe_result.has_value()) {
    return maybe_result.value();
  } else {
    throw mk_runtime_error(fmt::format(
        "ComputationGraphBuilder::get_broadcast_target_dims failed to find "
        "target tensor dims for input tensor dims {}",
        inputs_dims));
  }
}

tensor_guid_t ComputationGraphBuilder::dense(
    tensor_guid_t const &input,
    int outDim,
    std::optional<Activation> activation,
    bool use_bias,
    DataType data_type,
    std::optional<InitializerAttrs> const &projection_initializer,
    std::optional<InitializerAttrs> const &bias_initializer,
    std::optional<std::string> const &maybe_name,
    std::optional<std::string> const &projection_name,
    std::optional<std::string> const &bias_name) {
  LinearAttrs attrs =
      LinearAttrs{outDim, use_bias, data_type, activation, std::nullopt};

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};
  TensorShape output_shape =
      throw_if_unexpected(get_output_shape(attrs, this->get_shape(input)));

  std::vector<tensor_guid_t> weights;

  TensorShape projection_shape =
      throw_if_unexpected(get_projection_shape(attrs, this->get_shape(input)));

  tensor_guid_t projection_weights =
      this->create_weight(projection_shape,
                          CreateGrad::YES,
                          projection_initializer,
                          /*sync_type=*/std::nullopt,
                          projection_name);

  weights.push_back(projection_weights);

  if (use_bias) {
    TensorShape bias_shape =
        throw_if_unexpected(get_bias_shape(attrs, this->get_shape(input)));

    tensor_guid_t bias_weights = this->create_weight(bias_shape,
                                                     CreateGrad::YES,
                                                     bias_initializer,
                                                     /*sync_type=*/std::nullopt,
                                                     bias_name);
    weights.push_back(bias_weights);
  }

  return get_only(this->add_layer(
      layer, {input}, weights, {make_output_attrs(output_shape)}));
}

tensor_guid_t ComputationGraphBuilder::concat(
    std::vector<tensor_guid_t> const &inputs,
    int axis,
    std::optional<std::string> const &maybe_name) {

  ConcatAttrs attrs = ConcatAttrs{ff_dim_t{axis}};

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  std::vector<TensorShape> input_shapes = transform(
      inputs, [&](tensor_guid_t const &i) { return this->get_shape(i); });
  TensorShape output_shape =
      throw_if_unexpected(get_output_shape(attrs, input_shapes));

  return get_only(
    this->add_layer(layer, inputs, {}, {make_output_attrs(output_shape)})
  );
}

tensor_guid_t ComputationGraphBuilder::flat(tensor_guid_t const &input,
                                            int start_dim,
                                            std::optional<int> const &end_dim,
                                            std::optional<std::string> const &maybe_name) {
  int input_num_dims = num_dims(this->get_shape(input));

  FlatAttrs attrs = FlatAttrs{
    /*start_dim=*/ff_dim_t{start_dim},
    /*end_dim=*/ff_dim_t{end_dim.value_or(input_num_dims)},
  }; 

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  TensorShape output_shape = 
    get_output_shape(attrs, this->get_shape(input));

  return get_only(
    this->add_layer(layer, {input}, {}, {make_output_attrs(output_shape)})
  );
}

tensor_guid_t ComputationGraphBuilder::layer_norm(
    tensor_guid_t const &input,
    std::vector<int> const &axes,
    bool elementwise_affine,
    float eps,
    std::optional<std::string> const &maybe_name) {

  TensorShape input_shape = this->get_shape(input);

  if (any_of(axes,
             [&](size_t axis) { return axis >= num_dims(input_shape); })) {
    throw mk_runtime_error(fmt::format(
        "ComputationGraphBuilder::layer_norm received axes {} with "
        "out-of-bound element (input tensor has num dimensions = {})",
        axes,
        num_dims(input_shape)));
  }

  LayerNormAttrs attrs = LayerNormAttrs{
      stack_vector<ff_dim_t, MAX_TENSOR_DIM>{axes.begin(), axes.end()},
      elementwise_affine,
      eps,
  };

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};
  TensorShape output_shape =
      throw_if_unexpected(get_output_shape(attrs, input_shape));

  std::vector<TensorAttrs> weights;

  if (elementwise_affine) {
    // initializers chosen to match those of
    // https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm

    TensorShape gamma_shape =
        throw_if_unexpected(get_gamma_weights_shape(attrs, input_shape));
    InitializerAttrs gamma_initializer =
        InitializerAttrs{ConstantInitializerAttrs{DataTypeValue{float{1}}}};
    weights.push_back(make_weight_attrs(gamma_shape, gamma_initializer));

    TensorShape beta_shape =
        throw_if_unexpected(get_beta_weights_shape(attrs, input_shape));
    InitializerAttrs beta_initializer =
        InitializerAttrs{ConstantInitializerAttrs{DataTypeValue{float{0}}}};
    weights.push_back(make_weight_attrs(beta_shape, beta_initializer));
  }

  return get_only(this->add_layer(
      layer,
      {input},
      transform(weights,
                [&](TensorAttrs const &a) { return this->create_weight(a); }),
      {make_output_attrs(output_shape)}));
}

tensor_guid_t ComputationGraphBuilder::softmax(
    tensor_guid_t const &input,
    std::optional<int> maybe_dim,
    std::optional<std::string> const &maybe_name) {

  TensorShape input_shape = this->get_shape(input);

  int dim = maybe_dim.value_or(num_dims(input_shape) - 1);

  if (dim >= num_dims(input_shape)) {
    throw mk_runtime_error(
        fmt::format("ComputationGraphBuilder::softmax received out-of-bounds "
                    "dim {} for input tensor shape {}",
                    dim,
                    input_shape));
  }

  SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{dim}};

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};
  TensorShape output_shape =
      throw_if_unexpected(get_output_shape(attrs, input_shape));

  return get_only(
      this->add_layer(layer, {input}, {}, {make_output_attrs(output_shape)}));
}

} // namespace FlexFlow

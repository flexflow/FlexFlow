#include "pcg/computation_graph_builder.h"
#include "op-attrs/get_op_type.h"
#include "op-attrs/get_output_shapes.h"
#include "utils/expected.h"
#include "utils/fmt.h"

namespace FlexFlow {

Tensor ComputationGraphBuilder::as_type(Tensor const &x,
                                        DataType data_type,
                                        std::string const &name) {
  if (x.data_type < data_type) {
    return this->cast(x, data_type, name);
  } else if (x.data_type > data_type) {
    throw mk_runtime_error("Could not convert provided tensor data type {} to "
                           "desired data type {}",
                           x.data_type,
                           data_type);
  }
  return x;
}

static std::string get_default_name(OperatorType op_type) {
  return get_operator_type_name(op_type);
}

static std::string get_default_name(ComputationGraphAttrs const &attrs) {
  return get_default_name(get_op_type(attrs));
}

template <typename... Args>
static std::string get_default_name(variant<Args...> const &attrs) {
  return get_default_name(widen<ComputationGraphAttrs>(attrs));
}

Tensor ComputationGraphBuilder::element_unary(
    variant<ElementUnaryAttrs, ElementScalarUnaryAttrs> const &attrs,
    Tensor const &x,
    optional<std::string> const &maybe_name) {
  std::string name = maybe_name.value_or(get_default_name(attrs));

  Tensor input = this->as_type(x, DataType::FLOAT, name + "input_pre_cast");

  Layer layer = {widen<ComputationGraphAttrs>(attrs), name};
  TensorShape output_shape = get_output_shape(attrs, input);

  return this->add_layer(layer, {input}, {}, output_shape);
}

Tensor
    ComputationGraphBuilder::element_unary(OperatorType op_type,
                                           Tensor const &input,
                                           optional<std::string> const &name) {
  ElementUnaryAttrs attrs = {op_type};
  return this->element_unary(attrs, input, name);
}

Tensor ComputationGraphBuilder::element_scalar_unary(
    OperatorType op_type,
    Tensor const &input,
    float scalar,
    optional<std::string> const &name) {
  ElementScalarUnaryAttrs attrs = {op_type, scalar};
  return this->element_unary(attrs, input, name);
}

Tensor ComputationGraphBuilder::element_binary(
    OperatorType op_type,
    Tensor const &lhs,
    Tensor const &rhs,
    optional<std::string> const &maybe_name) {
  std::string name = maybe_name.value_or(get_default_name(op_type));

  TensorShape compute_shape = this->get_broadcast_target_shape({lhs, rhs});
  DataType compute_type = std::max(lhs.data_type, rhs.data_type);

  Tensor const lhs_input = this->as_type(this->broadcast(lhs, compute_shape),
                                         compute_type,
                                         name + "_inputl_pre_cast");
  Tensor const rhs_input = this->as_type(this->broadcast(rhs, compute_shape),
                                         compute_type,
                                         name + "_inputr_pre_cast");

  ElementBinaryAttrs attrs = {op_type, compute_type, false, false};

  Layer layer = {attrs, name};
  TensorShape output_shape = get_output_shape(attrs, lhs_input, rhs_input);

  return this->add_layer(layer, {lhs_input, rhs_input}, {}, output_shape);
}

Tensor ComputationGraphBuilder::exp(Tensor const &input,
                                    optional<std::string> const &name) {
  return this->element_unary(Op::EXP, input, name);
}

Tensor ComputationGraphBuilder::add(Tensor const &lhs,
                                    Tensor const &rhs,
                                    optional<std::string> const &name) {
  return this->element_binary(Op::EW_ADD, lhs, rhs, name);
}

Tensor ComputationGraphBuilder::subtract(Tensor const &lhs,
                                         Tensor const &rhs,
                                         optional<std::string> const &name) {
  return this->element_binary(Op::EW_SUB, lhs, rhs, name);
}

Tensor ComputationGraphBuilder::multiply(Tensor const &lhs,
                                         Tensor const &rhs,
                                         optional<std::string> const &name) {
  return this->element_binary(Op::EW_MUL, lhs, rhs, name);
}

Tensor ComputationGraphBuilder::divide(Tensor const &lhs,
                                       Tensor const &rhs,
                                       optional<std::string> const &name) {
  return this->element_binary(Op::EW_DIV, lhs, rhs, name);
}

Tensor ComputationGraphBuilder::max(Tensor const &lhs,
                                    Tensor const &rhs,
                                    optional<std::string> const &name) {
  return this->element_binary(Op::EW_MAX, lhs, rhs, name);
}

Tensor ComputationGraphBuilder::min(Tensor const &lhs,
                                    Tensor const &rhs,
                                    optional<std::string> const &name) {
  return this->element_binary(Op::EW_MIN, lhs, rhs, name);
}

Tensor ComputationGraphBuilder::rsqrt(Tensor const &input,
                                      optional<std::string> const &name) {
  return this->element_unary(Op::RSQRT, input, name);
}

Tensor ComputationGraphBuilder::pow(Tensor const &input,
                                    float exponent,
                                    optional<std::string> const &name) {
  return this->element_scalar_unary(Op::POW, input, exponent, name);
}

Tensor ComputationGraphBuilder::scalar_multiply(
    Tensor const &input, float scalar, optional<std::string> const &name) {
  return this->element_scalar_unary(Op::SCALAR_MULTIPLY, input, scalar, name);
}

Tensor ComputationGraphBuilder::scalar_add(Tensor const &input,
                                           float scalar,
                                           optional<std::string> const &name) {
  return this->element_scalar_unary(Op::SCALAR_ADD, input, scalar, name);
}

Tensor ComputationGraphBuilder::scalar_sub(Tensor const &lhs,
                                           float rhs,
                                           optional<std::string> const &name) {
  return this->element_scalar_unary(Op::SCALAR_SUB, lhs, rhs, name);
}

Tensor
    ComputationGraphBuilder::scalar_truediv(Tensor const &numerator,
                                            float denominator,
                                            optional<std::string> const &name) {
  return this->element_scalar_unary(
      Op::SCALAR_TRUE_DIV, numerator, denominator, name);
}

Tensor ComputationGraphBuilder::sin(Tensor const &input,
                                    optional<std::string> const &name) {
  return this->element_unary(Op::SIN, input, name);
}

Tensor ComputationGraphBuilder::cos(Tensor const &input,
                                    optional<std::string> const &name) {
  return this->element_unary(Op::COS, input, name);
}

Tensor ComputationGraphBuilder::relu(Tensor const &input,
                                     optional<std::string> const &name) {
  return this->element_unary(Op::RELU, input, name);
}

Tensor ComputationGraphBuilder::identity(Tensor const &input,
                                         optional<std::string> const &name) {
  return this->element_unary(Op::IDENTITY, input, name);
}

Tensor ComputationGraphBuilder::gelu(Tensor const &input,
                                     optional<std::string> const &name) {
  return this->element_unary(Op::GELU, input, name);
}

Tensor ComputationGraphBuilder::sigmoid(Tensor const &input,
                                        optional<std::string> const &name) {
  return this->element_unary(Op::SIGMOID, input, name);
}

Tensor ComputationGraphBuilder::tanh(Tensor const &input,
                                     optional<std::string> const &name) {
  return this->element_unary(Op::TANH, input, name);
}

Tensor ComputationGraphBuilder::elu(Tensor const &input,
                                    optional<std::string> const &name) {
  return this->element_unary(Op::ELU, input, name);
}

Tensor ComputationGraphBuilder::conv2d(
    Tensor const &x,
    int outChannels,
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int paddingH,
    int paddingW,
    optional<Activation> const &activation,
    int groups,
    bool use_bias,
    optional<Initializer const &> kernel_initializer,
    optional<Initializer const &> bias_initializer,
    optional<RegularizerAttrs const &> kernel_regularizer,
    optional<std::string> const &maybe_name) {
  Conv2DAttrs attrs = {outChannels,
                       kernelH,
                       kernelW,
                       strideH,
                       strideW,
                       paddingH,
                       paddingW,
                       groups,
                       activation,
                       use_bias};
  std::string name = maybe_name.value_or(get_default_name(attrs));

  Tensor input = this->as_type(x, DataType::FLOAT, name + "input_pre_cast");

  Layer layer = {attrs, name};
  TensorShape output_shape = get_output_shape(attrs, input);

  std::vector<std::pair<TensorShape, optional<Initializer>>> weights;

  weights.push_back({get_kernel_shape(attrs, input), kernel_initializer});

  if (use_bias) {
    weights.push_back({get_bias_shape(attrs, input), bias_initializer});
  }

  return this->add_layer(layer, {input}, weights, output_shape);
}

Tensor
    ComputationGraphBuilder::dropout(Tensor const &x,
                                     float rate,
                                     unsigned long long seed,
                                     optional<std::string> const &maybe_name) {
  DropoutAttrs attrs = {rate, seed};
  std::string name = maybe_name.value_or(get_default_name(attrs));

  Layer layer = {attrs, name};
  Tensor input = this->as_type(x, DataType::FLOAT, name + "input_pre_cast");

  TensorShape output_shape = get_output_shape(attrs, input);

  return this->add_layer(layer, {input}, {}, output_shape);
}

Tensor ComputationGraphBuilder::embedding(
    Tensor const &x,
    int num_entries,
    int outDim,
    AggregateOp aggr,
    DataType dtype,
    optional<Initializer const &> kernel_initializer,
    optional<std::string> const &maybe_name) {
  EmbeddingAttrs attrs = {num_entries, outDim, aggr, dtype};
  std::string name = maybe_name.value_or(get_default_name(attrs));

  Layer layer = {attrs, name};
  Tensor input = this->as_type(x, DataType::FLOAT, name + "input_pre_cast");

  TensorShape output_shape = get_output_shape(attrs, input);
  TensorShape weights_shape = get_weights_shape(attrs, input);

  return this->add_layer(
      layer, {input}, {{weights_shape, kernel_initializer}}, output_shape);
}

std::vector<Tensor>
    ComputationGraphBuilder::gather(Tensor const &input,
                                    Tensor const &index,
                                    ff_dim_t dim,
                                    optional<std::string> const &maybe_name) {
  GatherAttrs attrs = {dim};
  std::string name = maybe_name.value_or(get_default_name(attrs));

  Layer layer = {attrs, name};
  if (index.data_type != DataType::INT32 &&
      index.data_type != DataType::INT64) {
    throw mk_runtime_error("Invalid data type for input tensor 2 for Gather: "
                           "{} (should be {} or {})",
                           input.data_type,
                           DataType::INT32,
                           DataType::INT64);
  }
  std::vector<TensorShape> output_shapes =
      get_output_shapes(attrs, input, index);

  return this->add_layer(layer, {input}, {}, output_shapes);
}

TensorShape get_shape(Tensor const &);
std::vector<TensorShape> get_shape(std::vector<Tensor> const &);

Tensor ComputationGraphBuilder::aggregate(
    Tensor const &gate_preds,
    Tensor const &gate_assign,
    Tensor const &true_gate_assign,
    Tensor const &full_gate_gradients,
    std::vector<Tensor> const &exp_preds,
    int n,
    float lambda_bal,
    optional<std::string> const &maybe_name) {
  AggregateAttrs attrs = {n, lambda_bal};
  std::string name = maybe_name.value_or(get_default_name(attrs));

  Layer layer = {attrs, name};
  TensorShape output_shape = get_output_shape(attrs,
                                              get_shape(gate_preds),
                                              get_shape(gate_assign),
                                              get_shape(true_gate_assign),
                                              get_shape(full_gate_gradients),
                                              get_shape(exp_preds));

  std::vector<Tensor> inputs = {
      gate_preds, gate_assign, true_gate_assign, full_gate_gradients};
  extend(inputs, exp_preds);
  return this->add_layer(layer, inputs, {}, output_shape);
}

Tensor ComputationGraphBuilder::batch_norm(
    Tensor const &input, bool relu, optional<std::string> const &maybe_name) {
  BatchNormAttrs attrs = BatchNormAttrs{relu};
  std::string name = maybe_name.value_or(get_default_name(attrs));

  Layer layer = {attrs, name};

  TensorShape output_shape = get_output_shape(attrs, get_shape(input));

  return this->add_layer(layer, {input}, {}, output_shape);
}

} // namespace FlexFlow

#include "pcg/computation_graph_builder.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/get_op_type.h"
#include "op-attrs/get_output_shapes.h"
#include "op-attrs/ops/attention.h"
#include "op-attrs/ops/batch_norm.h"
#include "op-attrs/ops/broadcast.h"
#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/ops/dropout.h"
#include "op-attrs/ops/element_binary.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/embedding.h"
#include "op-attrs/ops/gather.h"
#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/softmax.h"
#include "op-attrs/ops/weight_attrs.dtg.h"
#include "pcg/computation_graph.h"
#include "utils/containers/any_of.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/enumerate_vector.h"
#include "utils/containers/get_only.h"
#include "utils/containers/transform.h"
#include "utils/expected.h"
#include <fmt/format.h>

namespace FlexFlow {

ComputationGraphBuilder::ComputationGraphBuilder()
    : computation_graph(make_empty_computation_graph()) {}

TensorShape ComputationGraphBuilder::get_shape(tensor_guid_t const &t) const {
  return get_tensor_attrs(this->computation_graph, t).shape;
}

tensor_guid_t ComputationGraphBuilder::create_tensor(TensorShape const &shape,
                                                     CreateGrad create_grad) {
  TensorAttrs tensor_attrs =
      TensorAttrs{shape, std::nullopt, std::nullopt, create_grad};
  LayerAttrs layer_attrs = LayerAttrs{
      ComputationGraphOpAttrs{InputAttrs{}},
      std::nullopt,
  };

  return this->add_layer(layer_attrs, {}, {}, tensor_attrs);
}

std::vector<tensor_guid_t> ComputationGraphBuilder::add_layer(
    LayerAttrs const &layer,
    std::vector<tensor_guid_t> const &inputs,
    std::vector<TensorAttrs> const &weights,
    std::vector<TensorAttrs> const &outputs) {
  std::vector<DataflowOutput> raw_weight_tensors;
  for (auto const &kv : enumerate_vector(weights)) {
    int weight_idx = kv.first;
    TensorAttrs weight_tensor_attrs = kv.second;

    std::optional<std::string> weight_name =
        transform(layer.name, [&](std::string const &layer_name) {
          return fmt::format("{}.weights[{}]", layer_name, weight_idx);
        });
    LayerAttrs weight_layer_attrs = LayerAttrs{
        ComputationGraphOpAttrs{WeightAttrs{weight_tensor_attrs.shape}},
        weight_name,
    };
    std::vector<DataflowOutput> weight_layer_inputs = {};
    std::vector<TensorAttrs> weight_output_attrs = {weight_tensor_attrs};
    raw_weight_tensors.push_back(get_only(this->computation_graph.raw_graph
                                              .add_node(weight_layer_attrs,
                                                        weight_layer_inputs,
                                                        weight_output_attrs)
                                              .outputs));
  }

  std::vector<DataflowOutput> raw_inputs = transform(
      inputs, [](tensor_guid_t const &t) { return t.raw_graph_output; });
  std::vector<DataflowOutput> raw_outputs =
      this->computation_graph.raw_graph
          .add_node(
              layer, concat_vectors(raw_inputs, raw_weight_tensors), outputs)
          .outputs;
  return transform(raw_outputs,
                   [](DataflowOutput const &o) { return tensor_guid_t{o}; });
}

tensor_guid_t
    ComputationGraphBuilder::add_layer(LayerAttrs const &layer,
                                       std::vector<tensor_guid_t> const &inputs,
                                       std::vector<TensorAttrs> const &weights,
                                       TensorAttrs const &output) {
  std::vector<TensorAttrs> outputs = {output};
  return get_only(this->add_layer(layer, inputs, weights, outputs));
}

std::vector<tensor_guid_t> ComputationGraphBuilder::add_layer(
    LayerAttrs const &layer,
    std::vector<tensor_guid_t> const &inputs,
    std::vector<TensorAttrs> const &weights,
    std::vector<TensorShape> const &outputs) {
  return this->add_layer(
      layer, inputs, weights, transform(outputs, [](TensorShape const &s) {
        return TensorAttrs{s, std::nullopt, std::nullopt, CreateGrad::YES};
      }));
}

tensor_guid_t
    ComputationGraphBuilder::add_layer(LayerAttrs const &layer,
                                       std::vector<tensor_guid_t> const &inputs,
                                       std::vector<TensorAttrs> const &weights,
                                       TensorShape const &output) {
  return get_only(this->add_layer(
      layer, inputs, weights, std::vector<TensorShape>{output}));
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

tensor_guid_t
    ComputationGraphBuilder::broadcast(tensor_guid_t const &input,
                                       TensorShape const &target_shape,
                                       std::string const &name) {
  TensorShape input_shape = this->get_shape(input);
  if (!tensor_shape_is_broadcastable_to(input_shape, target_shape)) {
    throw mk_runtime_error(fmt::format(
        "Cannot broadcast input tensor of shape {} to target shape {}",
        input_shape,
        target_shape));
  }

  BroadcastAttrs attrs = BroadcastAttrs{target_shape.dims};

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};
  TensorShape output_shape =
      throw_if_unexpected(get_output_shape(attrs, input_shape));

  return this->add_layer(layer, {input}, {}, output_shape);
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

  return this->add_layer(layer, {input}, {}, output_shape);
}

tensor_guid_t ComputationGraphBuilder::element_binary(
    OperatorType op_type,
    tensor_guid_t const &lhs,
    tensor_guid_t const &rhs,
    std::optional<std::string> const &maybe_name) {
  std::string name = maybe_name.value_or(get_default_name(op_type));

  TensorShape compute_shape = this->get_broadcast_target_shape({lhs, rhs});
  DataType compute_type =
      std::max(this->get_shape(lhs).data_type, this->get_shape(rhs).data_type);

  tensor_guid_t lhs_input = this->as_type(
      this->broadcast(
          lhs, compute_shape, fmt::format("{}_inputl_broadcast", name)),
      compute_type,
      name + "_inputl_cast");
  tensor_guid_t rhs_input = this->as_type(
      this->broadcast(
          rhs, compute_shape, fmt::format("{}_inputr_broadcast", name)),
      compute_type,
      name + "_inputr_cast");

  ElementBinaryAttrs attrs =
      ElementBinaryAttrs{op_type, compute_type, false, false};

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  TensorShape output_shape = throw_if_unexpected(get_output_shape(
      attrs, this->get_shape(lhs_input), this->get_shape(rhs_input)));

  return this->add_layer(layer, {lhs_input, rhs_input}, {}, output_shape);
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

static TensorAttrs make_weight_attrs(
    TensorShape const &shape,
    std::optional<InitializerAttrs> const &initializer_attrs) {
  return TensorAttrs{shape, initializer_attrs, std::nullopt, CreateGrad::YES};
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

  return this->add_layer(layer, {input}, weights, output_shape);
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

  return this->add_layer(layer, {input}, {}, output_shape);
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

  return this->add_layer(layer, {input}, {weight_attrs}, output_shape);
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

  return this->add_layer(layer, {input}, {}, output_shape);
}

/* std::vector<TensorShape>
 * ComputationGraphBuilder::get_shapes(std::vector<tensor_guid_t> const &ts)
 * const { */
/*   return transform(ts, [&](tensor_guid_t const &t) { return
 * this->get_shape(t); }); */
/* } */

// tensor_guid_t ComputationGraphBuilder::aggregate(
//     tensor_guid_t const &gate_preds,
//     tensor_guid_t const &gate_assign,
//     tensor_guid_t const &true_gate_assign,
//     tensor_guid_t const &full_gate_gradients,
//     std::vector<tensor_guid_t> const &exp_preds,
//     int n,
//     float lambda_bal,
//     std::optional<std::string> const &maybe_name) {
//   AggregateAttrs attrs = {n, lambda_bal};
//   std::string name = maybe_name.value_or(get_default_name(attrs));

//   LayerAttrs layer = {attrs, name};
//   TensorShape output_shape = get_output_shape(attrs,
//                                              this->get_shape(gate_preds),
//                                              this->get_shape(gate_assign),
//                                              this->get_shape(true_gate_assign),
//                                              this->get_shape(full_gate_gradients),
//                                              this->get_shape(exp_preds));

//   std::vector<tensor_guid_t> inputs = {
//       gate_preds, gate_assign, true_gate_assign, full_gate_gradients};
//   extend(inputs, exp_preds);
//   return this->add_layer(layer, inputs, {}, output_shape);
// }

tensor_guid_t ComputationGraphBuilder::batch_norm(
    tensor_guid_t const &input,
    bool relu,
    std::optional<std::string> const &maybe_name) {
  BatchNormAttrs attrs = BatchNormAttrs{relu};
  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};

  TensorShape output_shape = get_output_shape(attrs, this->get_shape(input));

  return this->add_layer(layer, {input}, {}, output_shape);
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

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};
  TensorShape output_shape =
      throw_if_unexpected(get_output_shape(attrs,
                                           this->get_shape(query),
                                           this->get_shape(key),
                                           this->get_shape(value)));

  TensorShape weights_shape =
      throw_if_unexpected(get_weights_shape(attrs,
                                            this->get_shape(query),
                                            this->get_shape(key),
                                            this->get_shape(value)));
  TensorAttrs weight_attrs = make_weight_attrs(weights_shape, initializer);

  return this->add_layer(layer,
                         std::vector<tensor_guid_t>{query, key, value},
                         {weight_attrs},
                         output_shape);
}

TensorShape ComputationGraphBuilder::get_broadcast_target_shape(
    std::vector<tensor_guid_t> const &inputs) {
  std::vector<TensorShape> input_shapes = transform(
      inputs, [&](tensor_guid_t const &t) { return this->get_shape(t); });

  return this->get_broadcast_target_shape(input_shapes);
}

TensorShape ComputationGraphBuilder::get_broadcast_target_shape(
    std::vector<TensorShape> const &input_shapes) {
  std::optional<TensorShape> maybe_result =
      ::FlexFlow::get_broadcast_target_shape(unordered_set_of(input_shapes));

  if (maybe_result.has_value()) {
    return maybe_result.value();
  } else {
    throw mk_runtime_error(fmt::format(
        "ComputationGraphBuilder::get_broadcast_target_shape failed to find "
        "target tensor shape for input tensor shapes {}",
        input_shapes));
  }
}

tensor_guid_t ComputationGraphBuilder::dense(
    tensor_guid_t const &input,
    int outDim,
    std::optional<Activation> activation,
    bool use_bias,
    DataType data_type,
    std::optional<InitializerAttrs> const &kernel_initializer,
    std::optional<InitializerAttrs> const &bias_initializer,
    std::optional<std::string> const &maybe_name) {
  LinearAttrs attrs =
      LinearAttrs{outDim, use_bias, data_type, activation, std::nullopt};

  std::string name =
      maybe_name.value_or(get_default_name(ComputationGraphOpAttrs{attrs}));

  LayerAttrs layer = LayerAttrs{ComputationGraphOpAttrs{attrs}, name};
  TensorShape output_shape =
      throw_if_unexpected(get_output_shape(attrs, this->get_shape(input)));

  std::vector<FlexFlow::TensorAttrs> weights;
  TensorShape kernel_shape =
      throw_if_unexpected(get_kernel_shape(attrs, this->get_shape(input)));
  weights.push_back(make_weight_attrs(kernel_shape, kernel_initializer));

  if (use_bias) {
    TensorShape bias_shape =
        throw_if_unexpected(get_bias_shape(attrs, this->get_shape(input)));
    weights.push_back(make_weight_attrs(bias_shape, bias_initializer));
  }

  return this->add_layer(layer, {input}, weights, output_shape);
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
        InitializerAttrs{ConstantInitializerAttrs{float{1}}};
    weights.push_back(make_weight_attrs(gamma_shape, gamma_initializer));

    TensorShape beta_shape =
        throw_if_unexpected(get_beta_weights_shape(attrs, input_shape));
    InitializerAttrs beta_initializer =
        InitializerAttrs{ConstantInitializerAttrs{float{0}}};
    weights.push_back(make_weight_attrs(beta_shape, beta_initializer));
  }

  return this->add_layer(layer, {input}, weights, output_shape);
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

  return this->add_layer(layer, {input}, {}, output_shape);
}

tensor_guid_t ComputationGraphBuilder::concat(
    int n,
    std::vector<tensor_guid_t> const &tensors,
    int axis,
    std::optional<std::string> const &maybe_name) {
  // NOT_IMPLEMENTED
  tensor_guid_t dummy_output = tensors.at(0);
  return dummy_output;
}

} // namespace FlexFlow

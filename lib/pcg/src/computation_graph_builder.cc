#include "pcg/computation_graph_builder.h"
#include "op-attrs/get_op_type.h"
#include "op-attrs/get_output_shapes.h"
#include "pcg/computation_graph.h"
#include "pcg/create_grad.h"
#include "pcg/layer_guid_t.h"
#include "pcg/tensor_guid_t.h"
#include "utils/containers.h"
#include "utils/exception.h"
#include "utils/expected.h"
#include "utils/fmt.h"
#include "utils/graph/multidiedge.h"

namespace FlexFlow {

static layer_guid_t add_layer(ComputationGraph &cg,
                              Layer const &layer,
                              std::vector<tensor_guid_t> const &inputs,
                              std::vector<weight_guid_t> const &weights,
                              std::vector<Tensor> const &outputs) {
  return cg.fmap(
      [&](OutputLabelledMultiDiGraph<Layer, Tensor> &g) -> layer_guid_t {
        auto guid = layer_guid_t{g.add_node(layer)};
        for (int i = 0; i < inputs.size(); i++) {
          g.add_edge(inputs[i], cg.get_input_slot(guid, i));
        }
        for (int i = 0; i < weights.size(); i++) {
          g.add_edge(weights[i], cg.get_weight_slot(guid, i));
        }
        for (int i = 0; i < outputs.size(); i++) {
          g.add_output(cg.get_output_slot(guid, i), outputs[i]);
        }

        return guid;
      });
}

static layer_guid_t
    add_layer(ComputationGraph &cg,
              Layer const &layer,
              std::vector<tensor_guid_t> const &inputs,
              std::vector<std::pair<TensorShape, optional<Initializer>>> const
                  &weight_shapes,
              std::vector<TensorShape> const &output_shapes) {
  std::vector<weight_guid_t> weights =
      transform(weight_shapes,
                [&](std::pair<TensorShape, optional<Initializer>> const &kv) {
                  return insert_new_weight_tensor(cg, kv.first, kv.second);
                });
  std::vector<Tensor> outputs =
      transform(output_shapes, [](TensorShape const &s) -> Tensor {
        return {s, CreateGrad::YES, nullopt, nullopt};
      });
  return add_layer(cg, layer, inputs, weights, outputs);
}

static std::vector<tensor_guid_t> get_output_tensors(ComputationGraph const &cg,
                                                     layer_guid_t layer) {
  std::unordered_set<MultiDiOutput> unsorted_outputs = get_outputs(cg, layer);
  std::vector<MultiDiOutput> outputs{unsorted_outputs.cbegin(),
                                     unsorted_outputs.cend()};
  return transform(sorted_by(outputs,
                             [&](MultiDiOutput const &o) {
                               return cg.output_for_port(o.idx);
                             }),
                   [](MultiDiOutput const &o) { return tensor_guid_t{o}; });
}

static tensor_guid_t get_only_output_tensor(ComputationGraph const &cg,
                                            layer_guid_t layer) {
  std::vector<tensor_guid_t> outputs = get_output_tensors(cg, layer);
  return get_only(outputs);
}

static std::vector<tensor_guid_t>
    insert_layer(ComputationGraph &cg,
                 Layer const &layer,
                 std::vector<tensor_guid_t> const &inputs,
                 std::vector<weight_guid_t> const &weights,
                 std::vector<Tensor> const &outputs) {
  return get_output_tensors(cg, add_layer(cg, layer, inputs, weights, outputs));
}

static std::vector<tensor_guid_t> insert_layer(
    ComputationGraph &cg,
    Layer const &layer,
    std::vector<tensor_guid_t> const &inputs,
    std::vector<std::pair<TensorShape, optional<Initializer>>> const &weights,
    std::vector<TensorShape> const &output_shapes) {
  return get_output_tensors(
      cg, add_layer(cg, layer, inputs, weights, output_shapes));
}

static tensor_guid_t insert_layer(ComputationGraph &cg,
                                  Layer const &layer,
                                  std::vector<tensor_guid_t> const &inputs,
                                  std::vector<weight_guid_t> const &weights,
                                  Tensor const &output) {
  return get_only_output_tensor(
      cg, add_layer(cg, layer, inputs, weights, {output}));
}

static tensor_guid_t insert_layer(
    ComputationGraph &cg,
    Layer const &layer,
    std::vector<tensor_guid_t> const &inputs,
    std::vector<std::pair<TensorShape, optional<Initializer>>> const &weights,
    TensorShape const &output_shape) {
  return get_only_output_tensor(
      cg, add_layer(cg, layer, inputs, weights, {output_shape}));
}

static TensorShape get_broadcast_target_shape(std::vector<TensorShape> const &);

static tensor_guid_t
    element_binary(ComputationGraph &,
                   OperatorType,
                   tensor_guid_t const &lhs,
                   tensor_guid_t const &rhs,
                   optional<std::string> const &name = nullopt);

static tensor_guid_t element_unary(ComputationGraph &,
                                   OperatorType,
                                   tensor_guid_t const &input,
                                   optional<std::string> const &name = nullopt);
static tensor_guid_t
    element_scalar_unary(ComputationGraph &,
                         OperatorType,
                         tensor_guid_t const &input,
                         float scalar,
                         optional<std::string> const &name = nullopt);
static tensor_guid_t
    element_unary(ComputationGraph &,
                  variant<ElementUnaryAttrs, ElementScalarUnaryAttrs> const &,
                  tensor_guid_t const &input,
                  optional<std::string> const &name = nullopt);

static tensor_guid_t as_type(ComputationGraph &cg,
                             tensor_guid_t const &x,
                             DataType data_type,
                             std::string const &name) {
  Tensor tensor = cg.at(x);
  if (tensor.get_data_type() < data_type) {
    return insert_cast_layer(cg, x, data_type, name);
  } else if (tensor.get_data_type() > data_type) {
    throw mk_runtime_error("Could not convert provided tensor data type {} to "
                           "desired data type {}",
                           tensor.get_data_type(),
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

static tensor_guid_t insert_element_unary_layer(
    ComputationGraph &cg,
    variant<ElementUnaryAttrs, ElementScalarUnaryAttrs> const &attrs,
    tensor_guid_t const &x,
    optional<std::string> const &maybe_name) {
  std::string name = maybe_name.value_or(get_default_name(attrs));

  tensor_guid_t input =
      as_type(cg, x, DataType::FLOAT, name + "input_pre_cast");

  Layer layer = {widen<ComputationGraphAttrs>(attrs), name};
  TensorShape output_shape = get_output_shape(attrs, input);

  return insert_layer(cg, layer, {input}, {}, output_shape);
}

static tensor_guid_t
    insert_element_unary_layer(ComputationGraph &,
                               OperatorType op_type,
                               tensor_guid_t const &input,
                               optional<std::string> const &name) {
  ElementUnaryAttrs attrs = {op_type};
  return insert_element_unary_layer(attrs, input, name);
}

static tensor_guid_t
    insert_element_scalar_unary_layer(ComputationGraph &,
                                      OperatorType op_type,
                                      tensor_guid_t const &input,
                                      float scalar,
                                      optional<std::string> const &name) {
  ElementScalarUnaryAttrs attrs = {op_type, scalar};
  return insert_element_unary_layer(attrs, input, name);
}

static tensor_guid_t
    insert_element_binary_layer(ComputationGraph &cg,
                                OperatorType op_type,
                                tensor_guid_t const &lhs,
                                tensor_guid_t const &rhs,
                                optional<std::string> const &maybe_name) {
  std::string name = maybe_name.value_or(get_default_name(op_type));

  TensorShape compute_shape = get_broadcast_target_shape({lhs, rhs});
  DataType compute_type = std::max(get_data_type(lhs), get_data_type(rhs));

  tensor_guid_t const lhs_input = as_type(
      broadcast(lhs, compute_shape), compute_type, name + "_inputl_pre_cast");
  tensor_guid_t const rhs_input = as_type(
      broadcast(rhs, compute_shape), compute_type, name + "_inputr_pre_cast");

  ElementBinaryAttrs attrs = {op_type, compute_type, false, false};

  Layer layer = {attrs, name};
  TensorShape output_shape = get_output_shape(attrs, lhs_input, rhs_input);

  return insert_layer(cg, layer, {lhs_input, rhs_input}, {}, output_shape);
}

tensor_guid_t insert_exp_layer(ComputationGraph &cg,
                               tensor_guid_t const &input,
                               optional<std::string> const &name) {
  return element_unary(cg, Op::EXP, input, name);
}

tensor_guid_t insert_add_layer(ComputationGraph &cg,
                               tensor_guid_t const &lhs,
                               tensor_guid_t const &rhs,
                               optional<std::string> const &name) {
  return element_binary(cg, Op::EW_ADD, lhs, rhs, name);
}

tensor_guid_t insert_subtract_layer(ComputationGraph &cg,
                                    tensor_guid_t const &lhs,
                                    tensor_guid_t const &rhs,
                                    optional<std::string> const &name) {
  return element_binary(cg, Op::EW_SUB, lhs, rhs, name);
}

tensor_guid_t insert_multiply_layer(ComputationGraph &cg,
                                    tensor_guid_t const &lhs,
                                    tensor_guid_t const &rhs,
                                    optional<std::string> const &name) {
  return element_binary(cg, Op::EW_MUL, lhs, rhs, name);
}

tensor_guid_t insert_divide_layer(ComputationGraph &cg,
                                  tensor_guid_t const &lhs,
                                  tensor_guid_t const &rhs,
                                  optional<std::string> const &name) {
  return element_binary(cg, Op::EW_DIV, lhs, rhs, name);
}

tensor_guid_t insert_max_layer(ComputationGraph &cg,
                               tensor_guid_t const &lhs,
                               tensor_guid_t const &rhs,
                               optional<std::string> const &name) {
  return element_binary(cg, Op::EW_MAX, lhs, rhs, name);
}

tensor_guid_t insert_min_layer(ComputationGraph &cg,
                               tensor_guid_t const &lhs,
                               tensor_guid_t const &rhs,
                               optional<std::string> const &name) {
  return element_binary(cg, Op::EW_MIN, lhs, rhs, name);
}

tensor_guid_t insert_rsqrt_layer(ComputationGraph &cg,
                                 tensor_guid_t const &input,
                                 optional<std::string> const &name) {
  return element_unary(cg, Op::RSQRT, input, name);
}

tensor_guid_t insert_pow_layer(ComputationGraph &cg,
                               tensor_guid_t const &input,
                               float exponent,
                               optional<std::string> const &name) {
  return element_scalar_unary(cg, Op::POW, input, exponent, name);
}

tensor_guid_t insert_scalar_multiply_layer(ComputationGraph &cg,
                                           tensor_guid_t const &input,
                                           float scalar,
                                           optional<std::string> const &name) {
  return element_scalar_unary(cg, Op::SCALAR_MULTIPLY, input, scalar, name);
}

tensor_guid_t insert_scalar_add_layer(ComputationGraph &cg,
                                      tensor_guid_t const &input,
                                      float scalar,
                                      optional<std::string> const &name) {
  return element_scalar_unary(cg, Op::SCALAR_ADD, input, scalar, name);
}

tensor_guid_t insert_scalar_sub_layer(ComputationGraph &cg,
                                      tensor_guid_t const &lhs,
                                      float rhs,
                                      optional<std::string> const &name) {
  return element_scalar_unary(cg, Op::SCALAR_SUB, lhs, rhs, name);
}

tensor_guid_t insert_scalar_truediv_layer(ComputationGraph &cg,
                                          tensor_guid_t const &numerator,
                                          float denominator,
                                          optional<std::string> const &name) {
  return element_scalar_unary(
      cg, Op::SCALAR_TRUE_DIV, numerator, denominator, name);
}

tensor_guid_t insert_sin_layer(ComputationGraph &cg,
                               tensor_guid_t const &input,
                               optional<std::string> const &name) {
  return element_unary(cg, Op::SIN, input, name);
}

tensor_guid_t insert_cos_layer(ComputationGraph &cg,
                               tensor_guid_t const &input,
                               optional<std::string> const &name) {
  return element_unary(cg, Op::COS, input, name);
}

tensor_guid_t insert_relu_layer(ComputationGraph &cg,
                                tensor_guid_t const &input,
                                optional<std::string> const &name) {
  return element_unary(cg, Op::RELU, input, name);
}

tensor_guid_t insert_identity_layer(ComputationGraph &cg,
                                    tensor_guid_t const &input,
                                    optional<std::string> const &name) {
  return element_unary(cg, Op::IDENTITY, input, name);
}

tensor_guid_t insert_gelu_layer(ComputationGraph &cg,
                                tensor_guid_t const &input,
                                optional<std::string> const &name) {
  return element_unary(cg, Op::GELU, input, name);
}

tensor_guid_t insert_sigmoid_layer(ComputationGraph &cg,
                                   tensor_guid_t const &input,
                                   optional<std::string> const &name) {
  return element_unary(cg, Op::SIGMOID, input, name);
}

tensor_guid_t insert_tanh_layer(ComputationGraph &cg,
                                tensor_guid_t const &input,
                                optional<std::string> const &name) {
  return element_unary(cg, Op::TANH, input, name);
}

tensor_guid_t insert_elu_layer(ComputationGraph &cg,
                               tensor_guid_t const &input,
                               optional<std::string> const &name) {
  return element_unary(cg, Op::ELU, input, name);
}

tensor_guid_t
    insert_conv2d_layer(ComputationGraph &cg,
                        tensor_guid_t const &x,
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

  tensor_guid_t input =
      as_type(cg, x, DataType::FLOAT, name + "input_pre_cast");

  Layer layer = {attrs, name};
  TensorShape output_shape = get_output_shape(attrs, input);

  std::vector<std::pair<TensorShape, optional<Initializer>>> weights;

  weights.push_back({get_kernel_shape(attrs, input), kernel_initializer});

  if (use_bias) {
    weights.push_back({get_bias_shape(attrs, input), bias_initializer});
  }

  return insert_layer(cg, layer, {input}, weights, output_shape);
}

tensor_guid_t insert_dropout_layer(ComputationGraph &cg,
                                   tensor_guid_t const &x,
                                   float rate,
                                   unsigned long long seed,
                                   optional<std::string> const &maybe_name) {
  DropoutAttrs attrs = {rate, seed};
  std::string name = maybe_name.value_or(get_default_name(attrs));

  Layer layer = {attrs, name};
  tensor_guid_t input =
      as_type(cg, x, DataType::FLOAT, name + "input_pre_cast");

  TensorShape output_shape = get_output_shape(attrs, input);

  return insert_layer(cg, layer, {input}, {}, output_shape);
}

tensor_guid_t
    insert_embedding_layer(ComputationGraph &cg,
                           tensor_guid_t const &x,
                           int num_entries,
                           int outDim,
                           AggregateOp aggr,
                           DataType dtype,
                           optional<Initializer const &> kernel_initializer,
                           optional<std::string> const &maybe_name) {
  EmbeddingAttrs attrs = {num_entries, outDim, aggr, dtype};
  std::string name = maybe_name.value_or(get_default_name(attrs));

  Layer layer = {attrs, name};
  tensor_guid_t input = as_type(x, DataType::FLOAT, name + "input_pre_cast");

  TensorShape output_shape = get_output_shape(attrs, input);
  TensorShape weights_shape = get_weights_shape(attrs, input);

  return insert_layer(
      cg, layer, {input}, {{weights_shape, kernel_initializer}}, output_shape);
}

std::vector<tensor_guid_t>
    insert_gather_layer(ComputationGraph &cg,
                        tensor_guid_t const &input,
                        tensor_guid_t const &index,
                        ff_dim_t dim,
                        optional<std::string> const &maybe_name) {
  GatherAttrs attrs = {dim};
  std::string name = maybe_name.value_or(get_default_name(attrs));

  Tensor index_tensor = cg.at(index);
  DataType index_dt = get_data_type(index_tensor);

  Layer layer = {attrs, name};
  if (index_dt != DataType::INT32 && index_dt != DataType::INT64) {
    throw mk_runtime_error("Invalid data type for input tensor 2 for Gather: "
                           "{} (should be {} or {})",
                           index_dt,
                           DataType::INT32,
                           DataType::INT64);
  }
  std::vector<TensorShape> output_shapes =
      get_output_shapes(attrs, input, index_tensor);

  return insert_layer(cg, layer, {input, index}, {}, output_shapes);
}

tensor_guid_t
    insert_aggregate_layer(ComputationGraph &cg,
                           tensor_guid_t const &gate_preds,
                           tensor_guid_t const &gate_assign,
                           tensor_guid_t const &true_gate_assign,
                           tensor_guid_t const &full_gate_gradients,
                           std::vector<tensor_guid_t> const &exp_preds,
                           int n,
                           float lambda_bal,
                           optional<std::string> const &maybe_name) {
  auto get_shape = [&](tensor_guid_t const &t) {
    return get_data_type(cg.at(t));
  };

  AggregateAttrs attrs = {n, lambda_bal};
  std::string name = maybe_name.value_or(get_default_name(attrs));

  Layer layer = {attrs, name};
  TensorShape output_shape = get_output_shape(attrs,
                                              get_shape(gate_preds),
                                              get_shape(gate_assign),
                                              get_shape(true_gate_assign),
                                              get_shape(full_gate_gradients),
                                              transform(exp_preds, get_shape));

  std::vector<tensor_guid_t> inputs = {
      gate_preds, gate_assign, true_gate_assign, full_gate_gradients};
  extend(inputs, exp_preds);
  return insert_layer(cg, layer, inputs, {}, output_shape);
}

tensor_guid_t insert_batch_norm_layer(ComputationGraph &cg,
                                      tensor_guid_t const &input,
                                      bool relu,
                                      optional<std::string> const &maybe_name) {
  BatchNormAttrs attrs = BatchNormAttrs{relu};
  std::string name = maybe_name.value_or(get_default_name(attrs));

  Layer layer = {attrs, name};

  TensorShape output_shape = get_output_shape(attrs, get_shape(input));

  return insert_layer(cg, layer, {input}, {}, output_shape);
}

} // namespace FlexFlow

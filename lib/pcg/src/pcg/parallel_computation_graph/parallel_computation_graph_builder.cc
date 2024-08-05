#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "op-attrs/ops/weight_attrs.dtg.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "pcg/parallel_computation_graph/generate_weight_transform.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/enumerate_vector.h"
#include "utils/containers/get_only.h"
#include "utils/containers/transform.h"
#include "op-attrs/parallel_op_attrs.h"

namespace FlexFlow {

static std::string get_default_name(OperatorType op_type) {
  return get_operator_type_name(op_type);
}

static std::string get_default_name(PCGOperatorAttrs const &attrs) {
  return get_default_name(get_op_type(attrs));
}

static ParallelTensorAttrs make_weight_attrs(
    ParallelTensorShape const &shape,
    std::optional<InitializerAttrs> const &initializer_attrs) {
  return ParallelTensorAttrs{
      /*shape=*/shape,
      /*sync_type=*/std::nullopt,
      /*initializer=*/initializer_attrs,
      /*create_gradients=*/CreateGrad::YES,
  };
}

ParallelComputationGraphBuilder::ParallelComputationGraphBuilder()
    : pcg(empty_parallel_computation_graph()) {}

parallel_tensor_guid_t ParallelComputationGraphBuilder::create_input_tensor(
    ParallelTensorShape const &shape,
    bool create_grad,
    std::optional<std::string> const &name) {
  ParallelTensorAttrs tensor_attrs = ParallelTensorAttrs{
      /*shape=*/shape,
      /*sync_type=*/std::nullopt,
      /*initializer=*/std::nullopt,
      /*create_gradients=*/(create_grad ? CreateGrad::YES : CreateGrad::NO),
  };
  ParallelLayerAttrs layer_attrs = ParallelLayerAttrs{
      PCGOperatorAttrs{InputAttrs{}},
      name,
  };

  return this->add_layer(layer_attrs, {}, {}, tensor_attrs);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::add(
    parallel_tensor_guid_t const &lhs,
    parallel_tensor_guid_t const &rhs,
    std::optional<std::string> const &maybe_name) {

  ParallelTensorShape lhs_shape = this->get_shape(lhs);
  ParallelTensorShape rhs_shape = this->get_shape(rhs);

  DataType datatype = [&] {
    if (lhs_shape.data_type != rhs_shape.data_type) {
      throw mk_runtime_error(
          fmt::format("Datatypes do not match: {} (lhs) != {} (rhs)",
                      lhs_shape.data_type,
                      rhs_shape.data_type));
    } else {
      return lhs_shape.data_type;
    }
  }();

  ElementBinaryAttrs attrs = ElementBinaryAttrs{
      OperatorType::EW_ADD,
      datatype,
      false,
      false,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};
  ParallelTensorShape output_shape =
      throw_if_unexpected(get_output_shape(attrs, lhs_shape, rhs_shape));

  return this->add_layer(layer, {lhs, rhs}, {}, output_shape);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::batch_matmul(
    parallel_tensor_guid_t const &a,
    parallel_tensor_guid_t const &b,
    std::optional<std::string> const &maybe_name) {

  BatchMatmulAttrs attrs = BatchMatmulAttrs{
      /*a_seq_length_dim=*/-1,
      /*b_seq_length_dim=*/-1,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};
  ParallelTensorShape output_shape = throw_if_unexpected(
      get_output_shape(attrs, this->get_shape(a), this->get_shape(b)));

  return this->add_layer(layer, {a, b}, {}, {output_shape});
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::cast(
    parallel_tensor_guid_t const &input,
    DataType result_type,
    std::optional<std::string> const &maybe_name) {

  CastAttrs attrs = CastAttrs{result_type};

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};
  ParallelTensorShape output_shape =
      throw_if_unexpected(get_output_shape(attrs, this->get_shape(input)));

  return this->add_layer(layer, {input}, {}, {output_shape});
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::conv2d(
    parallel_tensor_guid_t const &raw_input,
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
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  parallel_tensor_guid_t input =
      this->as_type(raw_input, DataType::FLOAT, name + "input_pre_cast");

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  ParallelTensorShape input_shape = this->get_shape(input);
  ParallelTensorShape output_shape = get_output_shape(attrs, input_shape);

  std::vector<ParallelTensorAttrs> weights;

  weights.push_back(make_weight_attrs(get_kernel_shape(attrs, input_shape),
                                      kernel_initializer));

  if (use_bias) {
    weights.push_back(make_weight_attrs(get_bias_shape(attrs, input_shape),
                                        bias_initializer));
  }

  return this->add_layer(layer, {input}, weights, output_shape);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::dense(
    parallel_tensor_guid_t const &input,
    int outDim,
    std::optional<Activation> activation,
    bool use_bias,
    DataType data_type,
    std::optional<InitializerAttrs> const &kernel_initializer,
    std::optional<InitializerAttrs> const &bias_initializer,
    std::optional<std::string> const &maybe_name) {
  LinearAttrs attrs = LinearAttrs{
      outDim,
      use_bias,
      data_type,
      activation,
      std::nullopt,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  ParallelTensorShape input_shape = this->get_shape(input);
  ParallelTensorShape output_shape =
      throw_if_unexpected(get_output_shape(attrs, input_shape));

  std::vector<ParallelTensorAttrs> weights;

  {
    ParallelTensorShape kernel_shape =
        throw_if_unexpected(get_kernel_shape(attrs, input_shape));
    weights.push_back(make_weight_attrs(kernel_shape, kernel_initializer));
  }

  if (use_bias) {
    ParallelTensorShape bias_shape =
        throw_if_unexpected(get_bias_shape(attrs, input_shape));
    weights.push_back(make_weight_attrs(bias_shape, bias_initializer));
  } else if (bias_initializer.has_value()) {
    throw mk_runtime_error("Dense received unexpected bias initializer even "
                           "though use_bias is set to false");
  }

  return this->add_layer(layer, {input}, weights, output_shape);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::embedding(
    parallel_tensor_guid_t const &input,
    int num_entries,
    int outDim,
    AggregateOp aggr,
    DataType dtype,
    std::optional<InitializerAttrs> const &kernel_initializer,
    std::optional<std::string> const &maybe_name) {

  EmbeddingAttrs attrs = EmbeddingAttrs{
      num_entries,
      outDim,
      aggr,
      dtype,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  ParallelTensorShape input_shape = this->get_shape(input);
  ParallelTensorShape output_shape =
      throw_if_unexpected(get_output_shape(attrs, input_shape));
  ParallelTensorShape weights_shape =
      throw_if_unexpected(get_weights_shape(attrs, input_shape));

  ParallelTensorAttrs weights_attrs =
      make_weight_attrs(weights_shape, kernel_initializer);

  return this->add_layer(layer, {input}, {weights_attrs}, output_shape);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::multihead_attention(
    parallel_tensor_guid_t const &query,
    parallel_tensor_guid_t const &key,
    parallel_tensor_guid_t const &value,
    int embed_dim,
    int num_heads,
    std::optional<int> maybe_kdim,
    std::optional<int> maybe_vdim,
    float dropout,
    bool bias,
    bool add_bias_kv,
    bool add_zero_attn,
    std::optional<InitializerAttrs> initializer,
    std::optional<InitializerAttrs> input_bias_initializer,
    std::optional<InitializerAttrs> output_bias_initializer,
    std::optional<std::string> const &maybe_name) {

  int kdim = maybe_kdim.value_or(embed_dim);
  int vdim = maybe_vdim.value_or(embed_dim);

  MultiHeadAttentionAttrs attrs = MultiHeadAttentionAttrs{
      /*embed_dim=*/embed_dim,
      /*num_heads=*/num_heads,
      /*kdim=*/kdim,
      /*vdim=*/vdim,
      /*dropout=*/dropout,
      /*bias=*/bias,
      /*add_bias_kv=*/add_bias_kv,
      /*add_zero_attn=*/add_zero_attn,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  ParallelTensorShape query_shape = this->get_shape(query);
  ParallelTensorShape key_shape = this->get_shape(key);
  ParallelTensorShape value_shape = this->get_shape(value);

  ParallelTensorShape output_shape = throw_if_unexpected(
      get_output_shape(attrs, query_shape, key_shape, value_shape));

  std::vector<ParallelTensorAttrs> weights;

  ParallelTensorAttrs weight_attrs = [&] {
    ParallelTensorShape weight_shape = throw_if_unexpected(
        get_weights_shape(attrs, query_shape, key_shape, value_shape));
    return make_weight_attrs(weight_shape, initializer);
  }();

  weights.push_back(weight_attrs);

  if (bias) {
    ParallelTensorShape input_bias_shape = throw_if_unexpected(
        get_input_bias_shape(attrs, query_shape, key_shape, value_shape));
    weights.push_back(
        make_weight_attrs(input_bias_shape, input_bias_initializer));
    ParallelTensorShape output_bias_shape = throw_if_unexpected(
        get_output_bias_shape(attrs, query_shape, key_shape, value_shape));
    weights.push_back(
        make_weight_attrs(output_bias_shape, output_bias_initializer));

  } else if (input_bias_initializer.has_value()) {
    throw mk_runtime_error("MultiheadAttention received unexpected input bias "
                           "initializer even though bias is set to false");
  } else if (output_bias_initializer.has_value()) {
    throw mk_runtime_error("MultiheadAttention received unexpected output bias "
                           "initializer even though bias is set to false");
  }

  return this->add_layer(layer, {query, key, value}, weights, output_shape);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::batch_norm(
    parallel_tensor_guid_t const &input,
    bool relu,
    std::optional<std::string> const &maybe_name) {
  
  BatchNormAttrs attrs = BatchNormAttrs{
    relu
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  ParallelTensorShape output_shape =
      get_output_shape(attrs, this->get_shape(input));

  return this->add_layer(layer, {input}, {}, {output_shape});
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::element_unary(
    ElementUnaryAttrs const &attrs,
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  ParallelTensorShape output_shape =
      throw_if_unexpected(get_output_shape(attrs, this->get_shape(input)));

  return this->add_layer(layer, {input}, {}, {output_shape});
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::relu(
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  ElementUnaryAttrs attrs = ElementUnaryAttrs{
    OperatorType::RELU,
    std::nullopt,
  };
   
  return this->element_unary(attrs, input, maybe_name);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::identity(
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  ElementUnaryAttrs attrs = ElementUnaryAttrs{
    OperatorType::IDENTITY,
    std::nullopt,
  };
   
  return this->element_unary(attrs, input, maybe_name);
}


parallel_tensor_guid_t ParallelComputationGraphBuilder::gelu(
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  ElementUnaryAttrs attrs = ElementUnaryAttrs{
    OperatorType::GELU,
    std::nullopt,
  };
   
  return this->element_unary(attrs, input, maybe_name);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::sigmoid(
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  ElementUnaryAttrs attrs = ElementUnaryAttrs{
    OperatorType::SIGMOID,
    std::nullopt,
  };
   
  return this->element_unary(attrs, input, maybe_name);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::tanh(
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  ElementUnaryAttrs attrs = ElementUnaryAttrs{
    OperatorType::TANH,
    std::nullopt,
  };
   
  return this->element_unary(attrs, input, maybe_name);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::elu(
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  ElementUnaryAttrs attrs = ElementUnaryAttrs{
    OperatorType::TANH,
    std::nullopt,
  };
   
  return this->element_unary(attrs, input, maybe_name);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::parallel_partition(
    parallel_tensor_guid_t const &input,
    ff_dim_t dim,
    int degree,
    std::optional<std::string> const &maybe_name) {

  RepartitionAttrs attrs = RepartitionAttrs{dim, degree};

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  ParallelTensorShape output_shape =
      throw_if_unexpected(get_output_shape(attrs, this->get_shape(input)));

  return this->add_layer(layer, {input}, {}, {output_shape});
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::parallel_combine(
    parallel_tensor_guid_t const &input,
    ff_dim_t dim,
    int degree,
    std::optional<std::string> const &maybe_name) {

  CombineAttrs attrs = CombineAttrs{dim, degree};

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  ParallelTensorShape output_shape =
      throw_if_unexpected(get_output_shape(attrs, this->get_shape(input)));

  return this->add_layer(layer, {input}, {}, {output_shape});
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::parallel_replicate(
    parallel_tensor_guid_t const &input,
    int degree,
    std::optional<std::string> const &maybe_name) {

  ReplicateAttrs attrs = ReplicateAttrs{degree};

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  ParallelTensorShape output_shape =
      get_output_shape(attrs, this->get_shape(input));

  return this->add_layer(layer, {input}, {}, {output_shape});
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::parallel_reduce(
    parallel_tensor_guid_t const &input,
    int degree,
    std::optional<std::string> const &maybe_name) {

  ReductionAttrs attrs = ReductionAttrs{degree};

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  ParallelTensorShape output_shape =
      throw_if_unexpected(get_output_shape(attrs, this->get_shape(input)));

  return this->add_layer(layer, {input}, {}, {output_shape});
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::as_type(
    parallel_tensor_guid_t const &input,
    DataType goal_datatype,
    std::string const &name) {
  DataType input_datatype = this->get_shape(input).data_type;
  if (input_datatype == goal_datatype) {
    return input;
  } else if (can_strictly_promote_datatype_from_to(input_datatype,
                                                   goal_datatype)) {
    return this->cast(input, goal_datatype, name);
  } else {
    throw mk_runtime_error(
        fmt::format("Could not convert provided tensor data type {} to "
                    "desired data type {}",
                    input_datatype,
                    goal_datatype));
  }
}

ParallelTensorShape ParallelComputationGraphBuilder::get_shape(
    parallel_tensor_guid_t const &t) const {
  return get_parallel_tensor_attrs(this->pcg, t).shape;
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::add_weight(ParallelTensorAttrs const &weight_tensor_attrs,
                                                                   std::optional<std::string> const &weight_name) {
  ParallelTensorShape par_weight_shape = weight_tensor_attrs.shape;
  TensorShape unpar_weight_shape = get_reduced_shape(weight_tensor_attrs.shape);

  ParallelLayerAttrs weight_layer_attrs = ParallelLayerAttrs{
      PCGOperatorAttrs{WeightAttrs{unpar_weight_shape}},
      weight_name,
  };

  std::vector<DataflowOutput> weight_layer_inputs = {};
  std::vector<ParallelTensorAttrs> weight_output_attrs = {
      weight_tensor_attrs};

  DataflowOutput current_raw_weight_tensor = get_only(this->pcg.raw_graph.add_node(weight_layer_attrs, weight_layer_inputs, weight_output_attrs).outputs);
  ParallelTensorShape current_shape = lift_to_parallel(unpar_weight_shape);

  for (ParallelOpAttrs const &parallel_op_attr : generate_weight_transform(unpar_weight_shape, par_weight_shape)) {
    ParallelTensorShape output_shape = get_output_shape(parallel_op_attr, current_shape);
    ParallelTensorAttrs output_attrs = ParallelTensorAttrs{
      output_shape,
      std::nullopt,
      std::nullopt,
      CreateGrad::YES,
    };
    
    ParallelLayerAttrs layer_attrs = ParallelLayerAttrs{
      pcg_op_attrs_from_parallel_op_attrs(parallel_op_attr),
      std::nullopt,
    };
    current_raw_weight_tensor = get_only(this->pcg.raw_graph.add_node(
                                 layer_attrs,
                                 {current_raw_weight_tensor},
                                 {output_attrs}).outputs);
    current_shape = output_shape;
  }

  assert (current_shape == par_weight_shape);

  return parallel_tensor_guid_t{current_raw_weight_tensor};
}

std::vector<parallel_tensor_guid_t> ParallelComputationGraphBuilder::add_layer(
    ParallelLayerAttrs const &layer,
    std::vector<parallel_tensor_guid_t> const &inputs,
    std::vector<ParallelTensorAttrs> const &weights,
    std::vector<ParallelTensorAttrs> const &outputs) {
  std::vector<DataflowOutput> raw_weight_tensors;
  for (auto const &kv : enumerate_vector(weights)) {
    int weight_idx = kv.first;
    ParallelTensorAttrs weight_tensor_attrs = kv.second;

    std::optional<std::string> weight_name =
        transform(layer.name, [&](std::string const &layer_name) {
          return fmt::format("{}.weights[{}]", layer_name, weight_idx);
        });

    raw_weight_tensors.push_back(this->add_weight(weight_tensor_attrs, weight_name).raw_graph_output);
  }

  std::vector<DataflowOutput> raw_inputs =
      transform(inputs, [](parallel_tensor_guid_t const &t) {
        return t.raw_graph_output;
      });
  std::vector<DataflowOutput> raw_outputs =
      this->pcg.raw_graph
          .add_node(
              layer, concat_vectors(raw_inputs, raw_weight_tensors), outputs)
          .outputs;
  return transform(raw_outputs, [](DataflowOutput const &o) {
    return parallel_tensor_guid_t{o};
  });
}

std::vector<parallel_tensor_guid_t> ParallelComputationGraphBuilder::add_layer(
    ParallelLayerAttrs const &layer,
    std::vector<parallel_tensor_guid_t> const &inputs,
    std::vector<ParallelTensorAttrs> const &weights,
    std::vector<ParallelTensorShape> const &outputs) {
  return this->add_layer(layer,
                         inputs,
                         weights,
                         transform(outputs, [](ParallelTensorShape const &s) {
                           return ParallelTensorAttrs{
                               /*shape=*/s,
                               /*sync_type=*/std::nullopt,
                               /*initializer=*/std::nullopt,
                               /*create_gradients=*/CreateGrad::YES,
                           };
                         }));
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::add_layer(
    ParallelLayerAttrs const &layer,
    std::vector<parallel_tensor_guid_t> const &inputs,
    std::vector<ParallelTensorAttrs> const &weights,
    ParallelTensorAttrs const &output) {
  std::vector<ParallelTensorAttrs> outputs = {output};
  return get_only(this->add_layer(layer, inputs, weights, outputs));
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::add_layer(
    ParallelLayerAttrs const &layer,
    std::vector<parallel_tensor_guid_t> const &inputs,
    std::vector<ParallelTensorAttrs> const &weights,
    ParallelTensorShape const &output) {
  std::vector<ParallelTensorShape> outputs = {output};
  return get_only(this->add_layer(layer, inputs, weights, outputs));
}

} // namespace FlexFlow

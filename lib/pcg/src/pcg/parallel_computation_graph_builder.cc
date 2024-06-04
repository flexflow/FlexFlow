#include "pcg/parallel_computation_graph_builder.h"
#include "op-attrs/ops/weight_attrs.dtg.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "pcg/parallel_computation_graph.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers.h"

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
  : pcg(empty_parallel_computation_graph()) { }

parallel_tensor_guid_t ParallelComputationGraphBuilder::create_input_tensor(ParallelTensorShape const &shape, 
                                                                            bool create_grad,
                                                                            std::optional<std::string> const &name) {
  ParallelTensorAttrs tensor_attrs = {
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

parallel_tensor_guid_t ParallelComputationGraphBuilder::add(parallel_tensor_guid_t const &lhs,
                           parallel_tensor_guid_t const &rhs,
                           std::optional<std::string> const &name) {
  NOT_IMPLEMENTED();
}

parallel_tensor_guid_t
ParallelComputationGraphBuilder::batch_matmul(parallel_tensor_guid_t const &a,
                                              parallel_tensor_guid_t const &b,
                                              /* int a_seq_length_dim = -1, */
                                              /* int b_seq_length_dim = -1, */
                                              std::optional<std::string> const &name) {
  NOT_IMPLEMENTED();
}

parallel_tensor_guid_t 
ParallelComputationGraphBuilder::cast(parallel_tensor_guid_t const &input,
                   DataType result_type,
                   std::optional<std::string> const &name) {
  NOT_IMPLEMENTED();
}

parallel_tensor_guid_t 
ParallelComputationGraphBuilder::conv2d(parallel_tensor_guid_t const &raw_input,
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

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  parallel_tensor_guid_t input =
      this->as_type(raw_input, DataType::FLOAT, name + "input_pre_cast");

  ParallelLayerAttrs layer = {PCGOperatorAttrs{attrs}, name};

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

parallel_tensor_guid_t 
ParallelComputationGraphBuilder::dense(parallel_tensor_guid_t const &input,
                                       int outDim,
                                       std::optional<Activation> activation,
                                       bool use_bias,
                                       DataType data_type,
                                       std::optional<InitializerAttrs> const &kernel_initializer,
                                       std::optional<InitializerAttrs> const &bias_initializer,
                                       std::optional<std::string> const &name) {
  NOT_IMPLEMENTED();
}


parallel_tensor_guid_t 
ParallelComputationGraphBuilder::embedding(
    parallel_tensor_guid_t const &input,
    int num_entries,
    int outDim,
    AggregateOp aggr,
    DataType dtype,
    std::optional<InitializerAttrs> const &kernel_initializer,
    std::optional<std::string> const &name) {
  NOT_IMPLEMENTED();
}

parallel_tensor_guid_t 
ParallelComputationGraphBuilder::multihead_attention(
    parallel_tensor_guid_t const &query,
    parallel_tensor_guid_t const &key,
    parallel_tensor_guid_t const &value,
    int embed_dim,
    int num_heads,
    int kdim,
    int vdim,
    float dropout,
    bool bias,
    bool add_bias_kv,
    bool add_zero_attn,
    std::optional<InitializerAttrs> initializer,
    std::optional<std::string> const &name) {
  NOT_IMPLEMENTED();
}

parallel_tensor_guid_t 
ParallelComputationGraphBuilder::relu(parallel_tensor_guid_t const &input,
                                      std::optional<std::string> const &name) {
  NOT_IMPLEMENTED();
}

parallel_tensor_guid_t 
ParallelComputationGraphBuilder::parallel_partition(parallel_tensor_guid_t const &x,
                                   ff_dim_t dim, 
                                   int degree,
                                   std::optional<std::string> const &name) {
  NOT_IMPLEMENTED();
}

parallel_tensor_guid_t 
ParallelComputationGraphBuilder::parallel_combine(parallel_tensor_guid_t const &x,
                               ff_dim_t dim,
                               int degree, 
                               std::optional<std::string> const &name) {
  NOT_IMPLEMENTED();
}

parallel_tensor_guid_t 
ParallelComputationGraphBuilder::parallel_replicate(parallel_tensor_guid_t const &x,
                                                    int degree,
                                                    std::optional<std::string> const &name) {
  NOT_IMPLEMENTED();
}

parallel_tensor_guid_t 
ParallelComputationGraphBuilder::parallel_reduce(parallel_tensor_guid_t const &x,
                                       int degree,
                                       std::optional<std::string> const &name) {
  NOT_IMPLEMENTED();
}

parallel_tensor_guid_t 
ParallelComputationGraphBuilder::as_type(parallel_tensor_guid_t const &input, DataType goal_datatype, std::string const &name) {
  DataType input_datatype = this->get_shape(input).data_type;
  if (input_datatype == goal_datatype) {
    return input;
  } else if (can_strictly_promote_datatype_from_to(input_datatype, goal_datatype)) {
    return this->cast(input, goal_datatype, name);
  } else {
    throw mk_runtime_error(
        fmt::format("Could not convert provided tensor data type {} to "
                    "desired data type {}",
                    input_datatype,
                    goal_datatype));
  }
}

ParallelTensorShape
ParallelComputationGraphBuilder::get_shape(parallel_tensor_guid_t const &t) const {
  return get_parallel_tensor_attrs(this->pcg, t).shape;
}

std::vector<parallel_tensor_guid_t>
ParallelComputationGraphBuilder::add_layer(ParallelLayerAttrs const &layer,
                        std::vector<parallel_tensor_guid_t> const &inputs,
                        std::vector<ParallelTensorAttrs> const &weights,
                        std::vector<ParallelTensorAttrs> const &outputs) {
  std::vector<MultiDiOutput> raw_weight_tensors;
  for (auto const &kv : enumerate_vector(weights)) {
    int weight_idx = kv.first;
    ParallelTensorAttrs weight_tensor_attrs = kv.second;

    std::optional<std::string> weight_name =
        transform(layer.name, [&](std::string const &layer_name) {
          return fmt::format("{}.weights[{}]", layer_name, weight_idx);
        });
    ParallelLayerAttrs weight_layer_attrs = ParallelLayerAttrs{
        PCGOperatorAttrs{WeightAttrs{}},
        weight_name,
    };
    std::vector<MultiDiOutput> weight_layer_inputs = {};
    std::vector<ParallelTensorAttrs> weight_output_attrs = {weight_tensor_attrs};
    raw_weight_tensors.push_back(
        get_only(this->pcg.raw_graph.add_operator(
            weight_layer_attrs, weight_layer_inputs, weight_output_attrs)));
  }

  std::vector<MultiDiOutput> raw_inputs = transform(
      inputs, [](parallel_tensor_guid_t const &t) { return t.raw_graph_output; });
  std::vector<MultiDiOutput> raw_outputs =
      this->pcg.raw_graph.add_operator(
          layer, concat_vectors(raw_inputs, raw_weight_tensors), outputs);
  return transform(raw_outputs,
                   [](MultiDiOutput const &o) { return parallel_tensor_guid_t{o}; });
}

std::vector<parallel_tensor_guid_t> 
ParallelComputationGraphBuilder::add_layer(ParallelLayerAttrs const &layer,
                        std::vector<parallel_tensor_guid_t> const &inputs,
                        std::vector<ParallelTensorAttrs> const &weights,
                        std::vector<ParallelTensorShape> const &outputs) {
  return this->add_layer(
      layer, inputs, weights, transform(outputs, [](ParallelTensorShape const &s) {
        return ParallelTensorAttrs{
          /*shape=*/s, 
          /*sync_type=*/std::nullopt, 
          /*initializer=*/std::nullopt,
          /*create_gradients=*/CreateGrad::YES,
        };
      }));
}


parallel_tensor_guid_t 
ParallelComputationGraphBuilder::add_layer(ParallelLayerAttrs const &layer,
                        std::vector<parallel_tensor_guid_t> const &inputs,
                        std::vector<ParallelTensorAttrs> const &weights,
                        ParallelTensorAttrs const &output) {
  std::vector<ParallelTensorAttrs> outputs = {output};
  return get_only(this->add_layer(layer, inputs, weights, outputs));
}

parallel_tensor_guid_t 
ParallelComputationGraphBuilder::add_layer(ParallelLayerAttrs const &layer,
                        std::vector<parallel_tensor_guid_t> const &inputs,
                        std::vector<ParallelTensorAttrs> const &weights,
                        ParallelTensorShape const &output) {
  std::vector<ParallelTensorShape> outputs = {output};
  return get_only(this->add_layer(layer, inputs, weights, outputs));
}


} // namespace FlexFlow

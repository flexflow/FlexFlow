#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "op-attrs/get_incoming_tensor_roles.h"
#include "op-attrs/ops/attention.h"
#include "op-attrs/ops/attention_attrs.dtg.h"
#include "op-attrs/ops/batch_norm.h"
#include "op-attrs/ops/batch_norm_attrs.dtg.h"
#include "op-attrs/ops/cast_attrs.dtg.h"
#include "op-attrs/ops/combine_attrs.dtg.h"
#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/ops/conv_2d_attrs.dtg.h"
#include "op-attrs/ops/element_binary_attrs.dtg.h"
#include "op-attrs/ops/element_unary_attrs.dtg.h"
#include "op-attrs/ops/embedding.h"
#include "op-attrs/ops/embedding_attrs.dtg.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/linear_attrs.dtg.h"
#include "op-attrs/ops/reduction_attrs.dtg.h"
#include "op-attrs/ops/repartition_attrs.dtg.h"
#include "op-attrs/ops/replicate_attrs.dtg.h"
#include "op-attrs/ops/weight_attrs.dtg.h"
#include "op-attrs/parallel_op_attrs.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "op-attrs/shape_inference.h"
#include "pcg/parallel_computation_graph/generate_weight_transform.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/count.h"
#include "utils/containers/enumerate_vector.h"
#include "utils/containers/get_only.h"
#include "utils/containers/repeat_element.h"
#include "utils/containers/require_only_key.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip_values_strict_with.h"
#include "utils/containers/zip_with.h"
#include "utils/exception.h"

namespace FlexFlow {

static std::string get_default_name(OperatorType op_type) {
  return get_operator_type_name(op_type);
}

static std::string get_default_name(PCGOperatorAttrs const &attrs) {
  return get_default_name(pcg_op_attrs_get_op_type(attrs));
}

ParallelComputationGraphBuilder::ParallelComputationGraphBuilder()
    : pcg(empty_parallel_computation_graph()) {}

parallel_tensor_guid_t ParallelComputationGraphBuilder::create_input_tensor(
    TensorShape const &shape, std::optional<std::string> const &name) {

  ParallelLayerAttrs layer_attrs = ParallelLayerAttrs{
      PCGOperatorAttrs{InputAttrs{shape}},
      name,
  };

  return require_only_key(
      add_parallel_layer(this->pcg,
                         layer_attrs,
                         {},
                         {},
                         std::map<TensorSlotName, CreateGrad>{
                             {
                                 TensorSlotName::OUTPUT,
                                 CreateGrad::NO,
                             },
                         })
          .outputs,
      TensorSlotName::OUTPUT);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::add(
    parallel_tensor_guid_t const &lhs,
    parallel_tensor_guid_t const &rhs,
    std::optional<std::string> const &maybe_name) {

  ParallelTensorShape lhs_shape = this->get_shape(lhs);
  ParallelTensorShape rhs_shape = this->get_shape(rhs);

  DataType datatype = [&] {
    if (lhs_shape.data_type != rhs_shape.data_type) {
      PANIC(fmt::format("Datatypes do not match: {} (lhs) != {} (rhs)",
                        lhs_shape.data_type,
                        rhs_shape.data_type));
    } else {
      return lhs_shape.data_type;
    }
  }();

  ElementBinaryAttrs attrs = ElementBinaryAttrs{
      ElementBinaryOp::ADD,
      datatype,
      false,
      false,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

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

parallel_tensor_guid_t ParallelComputationGraphBuilder::cast(
    parallel_tensor_guid_t const &input,
    DataType result_type,
    std::optional<std::string> const &maybe_name) {

  CastAttrs attrs = CastAttrs{result_type};

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

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

parallel_tensor_guid_t ParallelComputationGraphBuilder::conv2d(
    parallel_tensor_guid_t const &raw_input,
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
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  parallel_tensor_guid_t input =
      this->as_type(raw_input, DataType::FLOAT, name + "input_pre_cast");

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  ParallelTensorShape input_shape = this->get_shape(input);

  std::map<TensorSlotName, InitializerAttrs> initializers =
      conv2d_get_initializers(attrs,
                              get_reduced_shape(input_shape),
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

parallel_tensor_guid_t ParallelComputationGraphBuilder::dense(
    parallel_tensor_guid_t const &input,
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
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  ParallelTensorShape input_shape = this->get_shape(input);

  std::map<TensorSlotName, InitializerAttrs> initializers =
      throw_if_unexpected(get_initializers(attrs,
                                           get_reduced_shape(input_shape),
                                           maybe_projection_initializer,
                                           maybe_bias_initializer));

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

parallel_tensor_guid_t ParallelComputationGraphBuilder::embedding(
    parallel_tensor_guid_t const &input,
    positive_int num_entries,
    positive_int outDim,
    AggregateOp aggr,
    DataType dtype,
    std::optional<InitializerAttrs> const &maybe_kernel_initializer,
    std::optional<std::string> const &maybe_name) {

  EmbeddingAttrs attrs = EmbeddingAttrs{
      /*num_entries=*/num_entries,
      /*out_channels=*/outDim,
      /*aggr=*/aggr,
      /*data_type=*/dtype,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  std::map<TensorSlotName, InitializerAttrs> initializers =
      get_initializers(attrs, maybe_kernel_initializer);

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

parallel_tensor_guid_t ParallelComputationGraphBuilder::multihead_attention(
    parallel_tensor_guid_t const &query,
    parallel_tensor_guid_t const &key,
    parallel_tensor_guid_t const &value,
    positive_int embed_dim,
    positive_int num_heads,
    std::optional<positive_int> maybe_kdim,
    std::optional<positive_int> maybe_vdim,
    float dropout,
    bool bias,
    bool add_bias_kv,
    bool add_zero_attn,
    std::optional<InitializerAttrs> maybe_weights_initializer,
    std::optional<InitializerAttrs> maybe_input_bias_initializer,
    std::optional<InitializerAttrs> maybe_output_bias_initializer,
    std::optional<std::string> const &maybe_name) {

  positive_int kdim = maybe_kdim.value_or(embed_dim);
  positive_int vdim = maybe_vdim.value_or(embed_dim);

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

  std::map<TensorSlotName, InitializerAttrs> initializers = throw_if_unexpected(
      get_initializers(attrs,
                       get_reduced_shape(this->get_shape(query)),
                       get_reduced_shape(this->get_shape(key)),
                       get_reduced_shape(this->get_shape(value)),
                       maybe_weights_initializer,
                       maybe_input_bias_initializer,
                       maybe_output_bias_initializer));

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::QUERY,
                                                  query,
                                              },
                                              {
                                                  TensorSlotName::KEY,
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

parallel_tensor_guid_t ParallelComputationGraphBuilder::batch_norm(
    parallel_tensor_guid_t const &input,
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
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  ParallelTensorShape input_shape = this->get_shape(input);

  std::vector<ParallelTensorAttrs> weights;

  std::map<TensorSlotName, InitializerAttrs> initializers =
      throw_if_unexpected(get_initializers(attrs));

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

parallel_tensor_guid_t ParallelComputationGraphBuilder::element_unary(
    ElementUnaryAttrs const &attrs,
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

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

parallel_tensor_guid_t ParallelComputationGraphBuilder::relu(
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  ElementUnaryAttrs attrs = ElementUnaryAttrs{
      ElementUnaryOp::RELU,
      std::nullopt,
  };

  return this->element_unary(attrs, input, maybe_name);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::identity(
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  ElementUnaryAttrs attrs = ElementUnaryAttrs{
      ElementUnaryOp::IDENTITY,
      std::nullopt,
  };

  return this->element_unary(attrs, input, maybe_name);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::gelu(
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  ElementUnaryAttrs attrs = ElementUnaryAttrs{
      ElementUnaryOp::GELU,
      std::nullopt,
  };

  return this->element_unary(attrs, input, maybe_name);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::sigmoid(
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  ElementUnaryAttrs attrs = ElementUnaryAttrs{
      ElementUnaryOp::SIGMOID,
      std::nullopt,
  };

  return this->element_unary(attrs, input, maybe_name);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::tanh(
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  ElementUnaryAttrs attrs = ElementUnaryAttrs{
      ElementUnaryOp::TANH,
      std::nullopt,
  };

  return this->element_unary(attrs, input, maybe_name);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::elu(
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  ElementUnaryAttrs attrs = ElementUnaryAttrs{
      ElementUnaryOp::ELU,
      std::nullopt,
  };

  return this->element_unary(attrs, input, maybe_name);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::silu(
    parallel_tensor_guid_t const &input,
    std::optional<std::string> const &maybe_name) {

  ElementUnaryAttrs attrs = ElementUnaryAttrs{
      ElementUnaryOp::SILU,
      std::nullopt,
  };

  return this->element_unary(attrs, input, maybe_name);
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::parallel_partition(
    parallel_tensor_guid_t const &input,
    ff_dim_t dim,
    int_ge_two degree,
    std::optional<std::string> const &maybe_name) {

  RepartitionAttrs attrs = RepartitionAttrs{
      /*repartition_dim=*/dim,
      /*repartition_degree=*/degree,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

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

parallel_tensor_guid_t ParallelComputationGraphBuilder::parallel_combine(
    parallel_tensor_guid_t const &input,
    ff_dim_t dim,
    int_ge_two degree,
    std::optional<std::string> const &maybe_name) {

  CombineAttrs attrs = CombineAttrs{
      /*combine_dim=*/dim,
      /*combine_degree=*/degree,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

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

parallel_tensor_guid_t ParallelComputationGraphBuilder::parallel_replicate(
    parallel_tensor_guid_t const &input,
    int_ge_two degree,
    std::optional<std::string> const &maybe_name) {

  ReplicateAttrs attrs = ReplicateAttrs{degree};

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

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

parallel_tensor_guid_t ParallelComputationGraphBuilder::parallel_reduce(
    parallel_tensor_guid_t const &input,
    int_ge_two degree,
    std::optional<std::string> const &maybe_name) {

  ReductionAttrs attrs = ReductionAttrs{degree};

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

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
    PANIC(fmt::format("Could not convert provided tensor data type {} to "
                      "desired data type {}",
                      input_datatype,
                      goal_datatype));
  }
}

ParallelTensorShape ParallelComputationGraphBuilder::get_shape(
    parallel_tensor_guid_t const &t) const {
  return get_parallel_tensor_attrs(this->pcg, t).shape;
}

parallel_tensor_guid_t ParallelComputationGraphBuilder::add_weight(
    ParallelTensorShape const &par_weight_shape,
    InitializerAttrs const &initializer,
    std::optional<std::string> const &weight_name) {
  TensorShape unpar_weight_shape = get_reduced_shape(par_weight_shape);

  ParallelLayerAttrs weight_layer_attrs = ParallelLayerAttrs{
      PCGOperatorAttrs{WeightAttrs{
          /*shape=*/unpar_weight_shape,
          /*initializer=*/initializer,
      }},
      weight_name,
  };

  parallel_tensor_guid_t current_weight_tensor = require_only_key(
      add_parallel_layer(this->pcg, weight_layer_attrs, {}, {}).outputs,
      TensorSlotName::OUTPUT);

  for (ParallelOpAttrs const &parallel_op_attr :
       generate_weight_transform(unpar_weight_shape, par_weight_shape)) {

    ParallelLayerAttrs layer_attrs = ParallelLayerAttrs{
        pcg_op_attrs_from_parallel_op_attrs(parallel_op_attr),
        std::nullopt,
    };
    current_weight_tensor =
        require_only_key(add_parallel_layer(this->pcg,
                                            layer_attrs,
                                            {
                                                {
                                                    TensorSlotName::INPUT,
                                                    current_weight_tensor,
                                                },
                                            },
                                            {})
                             .outputs,
                         TensorSlotName::OUTPUT);
  }

  return current_weight_tensor;
}

static void
    check_incoming_tensor_roles(ParallelLayerAttrs const &layer,
                                std::set<TensorSlotName> const &input_slots,
                                std::set<TensorSlotName> const &weight_slots) {
  std::map<TensorSlotName, IncomingTensorRole> correct =
      get_incoming_tensor_roles(layer.op_attrs);
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

std::map<TensorSlotName, parallel_tensor_guid_t>
    ParallelComputationGraphBuilder::add_layer(
        ParallelLayerAttrs const &layer,
        std::map<TensorSlotName, parallel_tensor_guid_t> const &inputs,
        std::map<TensorSlotName, InitializerAttrs> const &weight_initializers) {

  ASSERT(are_disjoint(keys(inputs), keys(weight_initializers)));
  check_incoming_tensor_roles(layer, keys(inputs), keys(weight_initializers));

  std::map<TensorSlotName, ParallelTensorShape> input_shapes =
      map_values(inputs, [&](parallel_tensor_guid_t const &i) {
        return this->get_shape(i);
      });

  std::map<TensorSlotName, ParallelTensorShape> weight_shapes =
      get_weight_shapes(layer.op_attrs, input_shapes);
  std::map<TensorSlotName, parallel_tensor_guid_t> weight_tensors =
      zip_values_strict_with(weight_shapes,
                             weight_initializers,
                             [&](ParallelTensorShape const &weight_shape,
                                 InitializerAttrs const &initializer) {
                               return this->add_weight(weight_shape,
                                                       initializer);
                             });

  return add_parallel_layer(this->pcg, layer, inputs, weight_tensors, {})
      .outputs;
}

} // namespace FlexFlow

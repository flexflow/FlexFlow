#include "substitutions/unity_substitution_set.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/operator_type.h"
#include "op-attrs/ops/attention.h"
#include "op-attrs/ops/combine.h"
#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/ops/element_binary.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/repartition.h"
#include "op-attrs/ops/replicate.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "substitutions/apply_substitution/apply_substitution.h"
#include "substitutions/open_parallel_tensor_guid_t.h"
#include "substitutions/pcg_pattern.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/substitution_builder.h"
#include "test/utils/doctest/check_kv.h"
#include "utils/containers/get_only.h"
#include "utils/containers/require_only_key.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

template <typename T>
static ParallelLayerAttrs make_layer_attrs(
    T const &op_attrs,
    std::optional<std::string> const &maybe_name = std::nullopt) {
  return ParallelLayerAttrs{
      /*op_attrs=*/PCGOperatorAttrs{op_attrs},
      /*name=*/maybe_name,
  };
};

parallel_tensor_guid_t
    get_single_output(ParallelLayerAddedResult const &added) {
  return require_only_key(added.outputs, TensorSlotName::OUTPUT);
}

parallel_tensor_guid_t add_single_output_layer(
    ParallelComputationGraph &pcg,
    ParallelLayerAttrs const &layer_attrs,
    std::map<TensorSlotName, parallel_tensor_guid_t> const &inputs,
    std::map<TensorSlotName, parallel_tensor_guid_t> const &weights,
    std::optional<std::map<TensorSlotName, CreateGrad>> const &outputs =
        std::nullopt) {

  return get_single_output(
      add_parallel_layer(pcg, layer_attrs, inputs, weights, outputs));
}

parallel_tensor_guid_t add_input_layer(ParallelComputationGraph &pcg,
                                       TensorShape const &tensor_shape) {

  return get_single_output(pcg_add_input_layer(pcg, tensor_shape));
}

parallel_tensor_guid_t add_weight_layer(ParallelComputationGraph &pcg,
                                        TensorShape const &tensor_shape) {

  WeightAttrs weight_attrs = WeightAttrs{
      /*tensor_shape=*/tensor_shape,
      /*initializer=*/InitializerAttrs{ZeroInitializerAttrs{}},
  };

  return add_single_output_layer(pcg, make_layer_attrs(weight_attrs), {}, {});
}

parallel_tensor_guid_t
    add_replicate_layer(ParallelComputationGraph &pcg,
                        positive_int degree,
                        parallel_tensor_guid_t const &t_input) {

  ReplicateAttrs replicate_attrs = ReplicateAttrs{
      /*replicate_degree=*/degree,
  };

  return add_single_output_layer(pcg,
                                 make_layer_attrs(replicate_attrs),
                                 {{TensorSlotName::INPUT, t_input}},
                                 {});
}

parallel_tensor_guid_t
    add_reduction_layer(ParallelComputationGraph &pcg,
                        positive_int degree,
                        parallel_tensor_guid_t const &t_input) {

  ReductionAttrs reduction_attrs = ReductionAttrs{
      /*reduction_degree=*/degree,
  };

  return add_single_output_layer(pcg,
                                 make_layer_attrs(reduction_attrs),
                                 {{TensorSlotName::INPUT, t_input}},
                                 {});
}

parallel_tensor_guid_t
    add_partition_layer(ParallelComputationGraph &pcg,
                        ff_dim_t dim,
                        positive_int degree,
                        parallel_tensor_guid_t const &t_input) {

  RepartitionAttrs partition_attrs = RepartitionAttrs{
      /*repartition_dim=*/dim,
      /*repartition_degree=*/degree,
  };

  return add_single_output_layer(pcg,
                                 make_layer_attrs(partition_attrs),
                                 {{TensorSlotName::INPUT, t_input}},
                                 {});
}

parallel_tensor_guid_t
    add_combine_layer(ParallelComputationGraph &pcg,
                      ff_dim_t dim,
                      positive_int degree,
                      parallel_tensor_guid_t const &t_input) {

  CombineAttrs partition_attrs = CombineAttrs{
      /*combine_dim=*/dim,
      /*combine_degree=*/degree,
  };

  return add_single_output_layer(pcg,
                                 make_layer_attrs(partition_attrs),
                                 {{TensorSlotName::INPUT, t_input}},
                                 {});
}

parallel_tensor_guid_t add_linear_layer(
    ParallelComputationGraph &pcg,
    LinearAttrs const &linear_attrs,
    parallel_tensor_guid_t const &t_input,
    parallel_tensor_guid_t const &t_weight,
    std::optional<parallel_tensor_guid_t> const &t_bias = std::nullopt,
    std::optional<std::string> const &name = std::nullopt) {

  ASSERT(t_bias.has_value() == linear_attrs.use_bias);

  std::map<TensorSlotName, parallel_tensor_guid_t> weights = {
      {TensorSlotName::WEIGHT, t_weight},
  };

  if (t_bias.has_value()) {
    weights.insert({TensorSlotName::BIAS, t_bias.value()});
  }

  return add_single_output_layer(pcg,
                                 make_layer_attrs(linear_attrs, name),
                                 {{TensorSlotName::INPUT, t_input}},
                                 weights);
}

parallel_tensor_guid_t
    add_attention_layer(ParallelComputationGraph &pcg,
                        MultiHeadAttentionAttrs const &attn_attrs,
                        parallel_tensor_guid_t const &t_query,
                        parallel_tensor_guid_t const &t_key,
                        parallel_tensor_guid_t const &t_value,
                        parallel_tensor_guid_t const &t_weights,
                        std::optional<std::string> const &name = std::nullopt) {

  return add_single_output_layer(pcg,
                                 make_layer_attrs(attn_attrs, name),
                                 {
                                     {TensorSlotName::QUERY, t_query},
                                     {TensorSlotName::KEY, t_key},
                                     {TensorSlotName::VALUE, t_value},
                                 },
                                 {{TensorSlotName::WEIGHT, t_weights}});
}

parallel_tensor_guid_t add_conv2d_layer(
    ParallelComputationGraph &pcg,
    Conv2DAttrs const &conv2d_attrs,
    parallel_tensor_guid_t const &t_input,
    parallel_tensor_guid_t const &t_filter,
    std::optional<parallel_tensor_guid_t> const &bias = std::nullopt,
    std::optional<std::string> const &name = std::nullopt) {

  ASSERT(bias.has_value() == conv2d_attrs.use_bias);

  std::map<TensorSlotName, parallel_tensor_guid_t> weights = {
      {TensorSlotName::FILTER, t_filter},
  };

  if (bias.has_value()) {
    weights.insert({TensorSlotName::BIAS, bias.value()});
  }

  return add_single_output_layer(pcg,
                                 make_layer_attrs(conv2d_attrs, name),
                                 {{TensorSlotName::INPUT, t_input}},
                                 weights);
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_unity_substitution_set") {
    MachineComputeSpecification machine_spec = MachineComputeSpecification{
        /*num_nodes=*/2_p,
        /*num_cpus_per_node=*/8_p,
        /*num_gpus_per_node=*/4_p,
    };

    std::vector<Substitution> result = get_unity_substitution_set(machine_spec);

    CHECK(result.size() == 248);
  }

  TEST_CASE("get_expanded_substitution_set") {
    MachineComputeSpecification machine_spec = MachineComputeSpecification{
        /*num_nodes=*/2_p,
        /*num_cpus_per_node=*/8_p,
        /*num_gpus_per_node=*/4_p,
    };

    std::vector<Substitution> result =
        get_expanded_substitution_set(machine_spec);

    CHECK(result.size() == 254);
  }

  TEST_CASE("create_replicate_linear_combine, use_bias = false") {
    positive_int num_dims = 2_p;
    positive_int degree = 2_p;
    std::string linear_match = "linear_match";

    Substitution sub = create_replicate_linear_combine(num_dims, degree, false);

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                10_p,
                12_p,
            },
        },
        DataType::FLOAT,
    };

    LinearAttrs linear_attrs = LinearAttrs{
        /*out_channels=*/12_p,
        /*use_bias=*/false,
        /*data_type=*/DataType::FLOAT,
        /*activation=*/std::nullopt,
        /*regularizer=*/std::nullopt,
    };

    ReplicateAttrs replicate_input_attrs = ReplicateAttrs{
        /*replicate_degree=*/degree,
    };

    TensorShape projection_weight_shape =
        throw_if_unexpected(get_projection_shape(linear_attrs, input_shape));

    RepartitionAttrs partition_projection_attrs = RepartitionAttrs{
        /*repartition_dim=*/ff_dim_t{1_n},
        /*repartition_degree=*/degree,
    };

    SubParallelComputationGraph original_pcg = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      parallel_tensor_guid_t t_input = add_input_layer(pcg, input_shape);

      parallel_tensor_guid_t t_projection_weight =
          add_weight_layer(pcg, projection_weight_shape);

      parallel_tensor_guid_t t_linear = add_linear_layer(pcg,
                                                         linear_attrs,
                                                         t_input,
                                                         t_projection_weight,
                                                         /*bias=*/std::nullopt,
                                                         linear_match);

      return sub_pcg_from_full_pcg(pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t match_layer =
          get_parallel_layer_by_name(original_pcg, linear_match);
      open_parallel_tensor_guid_t match_layer_input_activations =
          get_layer_inputs(original_pcg, match_layer).at(TensorSlotName::INPUT);
      open_parallel_tensor_guid_t match_layer_input_weights =
          get_layer_inputs(original_pcg, match_layer)
              .at(TensorSlotName::WEIGHT);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, match_layer},
          },
          std::map<PatternInput, open_parallel_tensor_guid_t>{
              {
                  PatternInput{KwargDataflowGraphInput{0}},
                  match_layer_input_activations,
              },
              {
                  PatternInput{KwargDataflowGraphInput{2}},
                  match_layer_input_weights,
              }},
      };
    }();

    SubParallelComputationGraph result =
        apply_substitution(original_pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      parallel_tensor_guid_t t_replicated_input =
          add_replicate_layer(pcg, degree, add_input_layer(pcg, input_shape));

      parallel_tensor_guid_t t_partitioned_projection_weight =
          add_partition_layer(pcg,
                              ff_dim_t{0_n},
                              degree,
                              add_weight_layer(pcg, projection_weight_shape));

      parallel_tensor_guid_t t_replicated_linear =
          add_linear_layer(pcg,
                           linear_attrs,
                           t_replicated_input,
                           t_partitioned_projection_weight);

      parallel_tensor_guid_t t_combine =
          add_combine_layer(pcg, ff_dim_t{1_n}, degree, t_replicated_linear);

      return sub_pcg_from_full_pcg(pcg);
    }();

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }

  TEST_CASE("create_replicate_linear_combine, use_bias = true") {
    positive_int num_dims = 2_p;
    positive_int degree = 2_p;
    std::string linear_match = "linear_match";

    Substitution sub = create_replicate_linear_combine(num_dims, degree, true);

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                10_p,
                12_p,
            },
        },
        DataType::FLOAT,
    };

    LinearAttrs linear_attrs = LinearAttrs{
        /*out_channels=*/12_p,
        /*use_bias=*/true,
        /*data_type=*/DataType::FLOAT,
        /*activation=*/std::nullopt,
        /*regularizer=*/std::nullopt,
    };

    TensorShape projection_weight_shape =
        throw_if_unexpected(get_projection_shape(linear_attrs, input_shape));

    TensorShape bias_shape =
        throw_if_unexpected(get_bias_shape(linear_attrs, input_shape));

    SubParallelComputationGraph original_pcg = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      parallel_tensor_guid_t t_input = add_input_layer(pcg, input_shape);

      parallel_tensor_guid_t t_projection_weight =
          add_weight_layer(pcg, projection_weight_shape);

      parallel_tensor_guid_t t_bias = add_weight_layer(pcg, bias_shape);

      parallel_tensor_guid_t t_linear = add_linear_layer(
          /*pcg=*/pcg,
          /*linear_attrs=*/linear_attrs,
          /*t_input=*/t_input,
          /*t_weight=*/t_projection_weight,
          /*t_bias=*/t_bias,
          /*name=*/linear_match);

      return sub_pcg_from_full_pcg(pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t match_layer =
          get_parallel_layer_by_name(original_pcg, linear_match);
      open_parallel_tensor_guid_t match_layer_input_activations =
          get_layer_inputs(original_pcg, match_layer).at(TensorSlotName::INPUT);
      open_parallel_tensor_guid_t match_layer_input_weights =
          get_layer_inputs(original_pcg, match_layer)
              .at(TensorSlotName::WEIGHT);
      open_parallel_tensor_guid_t match_layer_input_bias =
          get_layer_inputs(original_pcg, match_layer).at(TensorSlotName::BIAS);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, match_layer},
          },
          std::map<PatternInput, open_parallel_tensor_guid_t>{
              {
                  PatternInput{KwargDataflowGraphInput{0}},
                  match_layer_input_activations,
              },
              {
                  PatternInput{KwargDataflowGraphInput{2}},
                  match_layer_input_weights,
              },
              {
                  PatternInput{KwargDataflowGraphInput{4}},
                  match_layer_input_bias,
              }},
      };
    }();

    SubParallelComputationGraph result =
        apply_substitution(original_pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      parallel_tensor_guid_t t_replicated_input =
          add_replicate_layer(pcg, degree, add_input_layer(pcg, input_shape));

      parallel_tensor_guid_t t_partitioned_projection_weight =
          add_partition_layer(pcg,
                              ff_dim_t{0_n},
                              degree,
                              add_weight_layer(pcg, projection_weight_shape));

      parallel_tensor_guid_t t_partitioned_bias = add_partition_layer(
          pcg, ff_dim_t{0_n}, degree, add_weight_layer(pcg, bias_shape));

      parallel_tensor_guid_t t_replicated_linear =
          add_linear_layer(pcg,
                           linear_attrs,
                           t_replicated_input,
                           t_partitioned_projection_weight,
                           t_partitioned_bias);

      parallel_tensor_guid_t t_combine =
          add_combine_layer(pcg, ff_dim_t{1_n}, degree, t_replicated_linear);

      return sub_pcg_from_full_pcg(pcg);
    }();

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }

  TEST_CASE("create_partition_linear_combine, use_bias = false") {
    positive_int num_dims = 2_p;
    positive_int degree = 2_p;
    std::string linear_match = "linear_match";

    Substitution sub = create_partition_linear_combine(num_dims, degree, false);

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                10_p,
                12_p,
            },
        },
        DataType::FLOAT,
    };

    LinearAttrs linear_attrs = LinearAttrs{
        /*out_channels=*/12_p,
        /*use_bias=*/false,
        /*data_type=*/DataType::FLOAT,
        /*activation=*/std::nullopt,
        /*regularizer=*/std::nullopt,
    };

    TensorShape projection_weight_shape =
        throw_if_unexpected(get_projection_shape(linear_attrs, input_shape));

    SubParallelComputationGraph original_pcg = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      parallel_tensor_guid_t t_input = add_input_layer(pcg, input_shape);

      parallel_tensor_guid_t t_projection_weight =
          add_weight_layer(pcg, projection_weight_shape);

      parallel_tensor_guid_t t_linear = add_linear_layer(pcg,
                                                         linear_attrs,
                                                         t_input,
                                                         t_projection_weight,
                                                         /*bias=*/std::nullopt,
                                                         linear_match);

      return sub_pcg_from_full_pcg(pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t match_layer =
          get_parallel_layer_by_name(original_pcg, linear_match);
      open_parallel_tensor_guid_t match_layer_input_activations =
          get_layer_inputs(original_pcg, match_layer).at(TensorSlotName::INPUT);
      open_parallel_tensor_guid_t match_layer_input_weights =
          get_layer_inputs(original_pcg, match_layer)
              .at(TensorSlotName::WEIGHT);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, match_layer},
          },
          std::map<PatternInput, open_parallel_tensor_guid_t>{
              {
                  PatternInput{KwargDataflowGraphInput{0}},
                  match_layer_input_activations,
              },
              {
                  PatternInput{KwargDataflowGraphInput{2}},
                  match_layer_input_weights,
              }},
      };
    }();

    SubParallelComputationGraph result =
        apply_substitution(original_pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      parallel_tensor_guid_t t_partitioned_input = add_partition_layer(
          pcg, ff_dim_t{0_n}, degree, add_input_layer(pcg, input_shape));

      parallel_tensor_guid_t t_replicated_projection_weight =
          add_replicate_layer(
              pcg, degree, add_weight_layer(pcg, projection_weight_shape));

      parallel_tensor_guid_t t_partitioned_linear =
          add_linear_layer(pcg,
                           linear_attrs,
                           t_partitioned_input,
                           t_replicated_projection_weight);

      parallel_tensor_guid_t t_combine =
          add_combine_layer(pcg, ff_dim_t{0_n}, degree, t_partitioned_linear);

      return sub_pcg_from_full_pcg(pcg);
    }();

    CHECK_MESSAGE(sub_pcgs_are_isomorphic(result, correct),
                  check_kv("result", sub_pcg_as_dot(result)),
                  check_kv("correct", sub_pcg_as_dot(correct)));
  }

  TEST_CASE("create_partition_linear_combine, use_bias = true") {
    positive_int num_dims = 2_p;
    positive_int degree = 2_p;
    std::string linear_match = "linear_match";

    Substitution sub = create_partition_linear_combine(num_dims, degree, true);

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                10_p,
                12_p,
            },
        },
        DataType::FLOAT,
    };

    LinearAttrs linear_attrs = LinearAttrs{
        /*out_channels=*/12_p,
        /*use_bias=*/true,
        /*data_type=*/DataType::FLOAT,
        /*activation=*/std::nullopt,
        /*regularizer=*/std::nullopt,
    };

    TensorShape projection_weight_shape =
        throw_if_unexpected(get_projection_shape(linear_attrs, input_shape));

    TensorShape bias_shape =
        throw_if_unexpected(get_bias_shape(linear_attrs, input_shape));

    SubParallelComputationGraph original_pcg = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      parallel_tensor_guid_t t_input = add_input_layer(pcg, input_shape);
      parallel_tensor_guid_t t_projection_weight =
          add_weight_layer(pcg, projection_weight_shape);
      parallel_tensor_guid_t t_bias = add_weight_layer(pcg, bias_shape);

      parallel_tensor_guid_t t_linear = add_linear_layer(pcg,
                                                         linear_attrs,
                                                         t_input,
                                                         t_projection_weight,
                                                         t_bias,
                                                         linear_match);

      return sub_pcg_from_full_pcg(pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t match_layer =
          get_parallel_layer_by_name(original_pcg, linear_match);

      open_parallel_tensor_guid_t match_layer_input_activations =
          get_layer_inputs(original_pcg, match_layer).at(TensorSlotName::INPUT);
      open_parallel_tensor_guid_t match_layer_input_weights =
          get_layer_inputs(original_pcg, match_layer)
              .at(TensorSlotName::WEIGHT);
      open_parallel_tensor_guid_t match_layer_input_bias =
          get_layer_inputs(original_pcg, match_layer).at(TensorSlotName::BIAS);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, match_layer},
          },
          std::map<PatternInput, open_parallel_tensor_guid_t>{
              {
                  PatternInput{KwargDataflowGraphInput{0}},
                  match_layer_input_activations,
              },
              {
                  PatternInput{KwargDataflowGraphInput{2}},
                  match_layer_input_weights,
              },
              {
                  PatternInput{KwargDataflowGraphInput{4}},
                  match_layer_input_bias,
              }},
      };
    }();

    SubParallelComputationGraph result =
        apply_substitution(original_pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      parallel_tensor_guid_t t_partitioned_input = add_partition_layer(
          pcg, ff_dim_t{0_n}, degree, add_input_layer(pcg, input_shape));

      parallel_tensor_guid_t t_replicated_projection_weight =
          add_replicate_layer(
              pcg, degree, add_weight_layer(pcg, projection_weight_shape));

      parallel_tensor_guid_t t_replicated_bias =
          add_replicate_layer(pcg, degree, add_weight_layer(pcg, bias_shape));

      parallel_tensor_guid_t t_partitioned_linear =
          add_linear_layer(pcg,
                           linear_attrs,
                           t_partitioned_input,
                           t_replicated_projection_weight,
                           t_replicated_bias);

      parallel_tensor_guid_t t_combine =
          add_combine_layer(pcg, ff_dim_t{0_n}, degree, t_partitioned_linear);

      return sub_pcg_from_full_pcg(pcg);
    }();

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }

  TEST_CASE("create_partition_conv2d_combine") {
    positive_int outChannels = 6_p;
    positive_int kernelH = 5_p;
    positive_int kernelW = 4_p;
    positive_int strideH = 3_p;
    positive_int strideW = 2_p;
    nonnegative_int paddingH = 1_n;
    nonnegative_int paddingW = 0_n;
    positive_int num_dims = 4_p;
    positive_int degree = 2_p;
    std::string conv2d_match = "conv2d_match";

    Substitution sub = create_partition_conv2d_combine(num_dims, degree);

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                12_p,
                3_p,
                12_p,
                10_p,
            },
        },
        DataType::FLOAT,
    };

    Conv2DAttrs conv2d_attrs = Conv2DAttrs{
        /*outChannels=*/outChannels,
        /*kernelH=*/kernelH,
        /*kernelW=*/kernelW,
        /*strideH=*/strideH,
        /*strideW=*/strideW,
        /*paddingH=*/paddingH,
        /*paddingW=*/paddingW,
        /*groups=*/1_p,
        /*activation=*/std::nullopt,
        /*use_bias=*/false,
    };

    SubParallelComputationGraph original_pcg = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      parallel_tensor_guid_t t_input = add_input_layer(pcg, input_shape);

      TensorShape casted_input_shape =
          get_reduced_shape(get_parallel_tensor_shape(pcg, t_input));

      TensorShape projection_weight_shape =
          conv2d_get_weight_shapes(conv2d_attrs, casted_input_shape)
              .at(TensorSlotName::FILTER);

      parallel_tensor_guid_t t_projection_weight =
          add_weight_layer(pcg, projection_weight_shape);

      parallel_tensor_guid_t t_conv = add_conv2d_layer(pcg,
                                                       conv2d_attrs,
                                                       t_input,
                                                       t_projection_weight,
                                                       /*bias=*/std::nullopt,
                                                       conv2d_match);

      return sub_pcg_from_full_pcg(pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t match_layer =
          get_parallel_layer_by_name(original_pcg, conv2d_match);
      open_parallel_tensor_guid_t match_layer_input_activations =
          get_layer_inputs(original_pcg, match_layer).at(TensorSlotName::INPUT);
      open_parallel_tensor_guid_t match_layer_input_weights =
          get_layer_inputs(original_pcg, match_layer)
              .at(TensorSlotName::FILTER);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, match_layer},
          },
          std::map<PatternInput, open_parallel_tensor_guid_t>{
              {
                  PatternInput{KwargDataflowGraphInput{0}},
                  match_layer_input_activations,
              },
              {
                  PatternInput{KwargDataflowGraphInput{2}},
                  match_layer_input_weights,
              }},
      };
    }();

    SubParallelComputationGraph result =
        apply_substitution(original_pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      parallel_tensor_guid_t t_input = add_input_layer(pcg, input_shape);
      parallel_tensor_guid_t t_partitioned_input =
          add_partition_layer(pcg, ff_dim_t{0_n}, degree, t_input);

      TensorShape casted_input_shape =
          get_reduced_shape(get_parallel_tensor_shape(pcg, t_input));

      TensorShape weight_shape =
          conv2d_get_weight_shapes(conv2d_attrs, casted_input_shape)
              .at(TensorSlotName::FILTER);

      parallel_tensor_guid_t t_replicated_weight =
          add_replicate_layer(pcg, degree, add_weight_layer(pcg, weight_shape));

      parallel_tensor_guid_t t_partitioned_conv2d = add_conv2d_layer(
          pcg, conv2d_attrs, t_partitioned_input, t_replicated_weight);

      parallel_tensor_guid_t t_combine =
          add_combine_layer(pcg, ff_dim_t{0_n}, degree, t_partitioned_conv2d);

      return sub_pcg_from_full_pcg(pcg);
    }();

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }

  TEST_CASE("create_partition_attention_combine") {
    positive_int embed_dim = 8_p;
    positive_int num_heads = 6_p;
    positive_int degree = 2_p;
    std::string attention_match = "attention_match";

    Substitution sub = create_partition_attention_combine(num_heads, degree);

    TensorShape query_shape = TensorShape{
        TensorDims{
            FFOrdered{
                12_p,
                16_p,
                10_p,
            },
        },
        DataType::FLOAT,
    };
    TensorShape key_shape = query_shape;
    TensorShape value_shape = query_shape;

    MultiHeadAttentionAttrs attention_attrs = MultiHeadAttentionAttrs{
        /*embed_dim=*/embed_dim,
        /*num_heads=*/num_heads,
        /*kdim=*/embed_dim,
        /*vdim=*/embed_dim,
        /*dropout=*/0,
        /*bias=*/false,
        /*add_bias_kv=*/false,
        /*add_zero_attn=*/false,
    };

    TensorShape weights_shape = throw_if_unexpected(get_weights_shape(
        attention_attrs, query_shape, key_shape, value_shape));

    SubParallelComputationGraph original_pcg = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      parallel_tensor_guid_t t_query = add_input_layer(pcg, query_shape);
      parallel_tensor_guid_t t_key = add_input_layer(pcg, key_shape);
      parallel_tensor_guid_t t_value = add_input_layer(pcg, value_shape);

      parallel_tensor_guid_t t_weights = add_weight_layer(pcg, weights_shape);

      parallel_tensor_guid_t t_attention = add_attention_layer(pcg,
                                                               attention_attrs,
                                                               t_query,
                                                               t_key,
                                                               t_value,
                                                               t_weights,
                                                               attention_match);

      return sub_pcg_from_full_pcg(pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t match_layer =
          get_parallel_layer_by_name(original_pcg, attention_match);
      open_parallel_tensor_guid_t match_layer_query =
          get_layer_inputs(original_pcg, match_layer).at(TensorSlotName::QUERY);
      open_parallel_tensor_guid_t match_layer_key =
          get_layer_inputs(original_pcg, match_layer).at(TensorSlotName::KEY);
      open_parallel_tensor_guid_t match_layer_value =
          get_layer_inputs(original_pcg, match_layer).at(TensorSlotName::VALUE);
      open_parallel_tensor_guid_t match_layer_input_weights =
          get_layer_inputs(original_pcg, match_layer)
              .at(TensorSlotName::WEIGHT);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, match_layer},
          },
          std::map<PatternInput, open_parallel_tensor_guid_t>{
              {
                  PatternInput{KwargDataflowGraphInput{0}},
                  match_layer_query,
              },
              {
                  PatternInput{KwargDataflowGraphInput{2}},
                  match_layer_key,
              },
              {
                  PatternInput{KwargDataflowGraphInput{4}},
                  match_layer_value,
              },
              {
                  PatternInput{KwargDataflowGraphInput{6}},
                  match_layer_input_weights,
              }},
      };
    }();

    SubParallelComputationGraph result =
        apply_substitution(original_pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      parallel_tensor_guid_t t_query = add_partition_layer(
          pcg, ff_dim_t{0_n}, degree, add_input_layer(pcg, query_shape));
      parallel_tensor_guid_t t_key = add_partition_layer(
          pcg, ff_dim_t{0_n}, degree, add_input_layer(pcg, key_shape));
      parallel_tensor_guid_t t_value = add_partition_layer(
          pcg, ff_dim_t{0_n}, degree, add_input_layer(pcg, value_shape));

      parallel_tensor_guid_t t_weight = add_replicate_layer(
          pcg, degree, add_weight_layer(pcg, weights_shape));

      parallel_tensor_guid_t t_partitioned_attention = add_attention_layer(
          pcg, attention_attrs, t_query, t_key, t_value, t_weight);

      parallel_tensor_guid_t t_combine = add_combine_layer(
          pcg, ff_dim_t{0_n}, degree, t_partitioned_attention);

      return sub_pcg_from_full_pcg(pcg);
    }();

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }

  TEST_CASE("create_replicate_attention_reduce") {
    positive_int embed_dim = 8_p;
    positive_int num_heads = 6_p;
    positive_int degree = 2_p;
    std::string attention_match = "attention_match";

    Substitution sub = create_replicate_attention_reduce(num_heads, degree);

    TensorShape query_shape = TensorShape{
        TensorDims{
            FFOrdered{
                12_p,
                16_p,
                10_p,
            },
        },
        DataType::FLOAT,
    };
    TensorShape key_shape = query_shape;
    TensorShape value_shape = query_shape;

    MultiHeadAttentionAttrs attention_attrs = MultiHeadAttentionAttrs{
        /*embed_dim=*/embed_dim,
        /*num_heads=*/num_heads,
        /*kdim=*/embed_dim,
        /*vdim=*/embed_dim,
        /*dropout=*/0,
        /*bias=*/false,
        /*add_bias_kv=*/false,
        /*add_zero_attn=*/false,
    };

    TensorShape weight_shape = throw_if_unexpected(get_weights_shape(
        attention_attrs, query_shape, key_shape, value_shape));

    SubParallelComputationGraph original_pcg = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      parallel_tensor_guid_t t_query = add_input_layer(pcg, query_shape);
      parallel_tensor_guid_t t_key = add_input_layer(pcg, key_shape);
      parallel_tensor_guid_t t_value = add_input_layer(pcg, value_shape);

      parallel_tensor_guid_t t_weight = add_weight_layer(pcg, weight_shape);

      parallel_tensor_guid_t attention_added =
          add_attention_layer(pcg,
                              attention_attrs,
                              t_query,
                              t_key,
                              t_value,
                              t_weight,
                              attention_match);

      return sub_pcg_from_full_pcg(pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t match_layer =
          get_parallel_layer_by_name(original_pcg, attention_match);
      open_parallel_tensor_guid_t match_layer_query =
          get_layer_inputs(original_pcg, match_layer).at(TensorSlotName::QUERY);
      open_parallel_tensor_guid_t match_layer_key =
          get_layer_inputs(original_pcg, match_layer).at(TensorSlotName::KEY);
      open_parallel_tensor_guid_t match_layer_value =
          get_layer_inputs(original_pcg, match_layer).at(TensorSlotName::VALUE);
      open_parallel_tensor_guid_t match_layer_input_weights =
          get_layer_inputs(original_pcg, match_layer)
              .at(TensorSlotName::WEIGHT);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, match_layer},
          },
          std::map<PatternInput, open_parallel_tensor_guid_t>{
              {
                  PatternInput{KwargDataflowGraphInput{0}},
                  match_layer_query,
              },
              {
                  PatternInput{KwargDataflowGraphInput{2}},
                  match_layer_key,
              },
              {
                  PatternInput{KwargDataflowGraphInput{4}},
                  match_layer_value,
              },
              {
                  PatternInput{KwargDataflowGraphInput{6}},
                  match_layer_input_weights,
              }},
      };
    }();

    SubParallelComputationGraph result =
        apply_substitution(original_pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      parallel_tensor_guid_t t_query =
          add_replicate_layer(pcg, degree, add_input_layer(pcg, query_shape));
      parallel_tensor_guid_t t_key =
          add_replicate_layer(pcg, degree, add_input_layer(pcg, key_shape));
      parallel_tensor_guid_t t_value =
          add_replicate_layer(pcg, degree, add_input_layer(pcg, value_shape));

      parallel_tensor_guid_t t_weight = add_partition_layer(
          pcg, ff_dim_t{1_n}, degree, add_weight_layer(pcg, weight_shape));

      parallel_tensor_guid_t t_replicated_attention = add_attention_layer(
          pcg, attention_attrs, t_query, t_key, t_value, t_weight);

      parallel_tensor_guid_t t_reduction =
          add_reduction_layer(pcg, degree, t_replicated_attention);

      return sub_pcg_from_full_pcg(pcg);
    }();

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }

  TEST_CASE("create_partition_softmax_combine") {
    positive_int degree = 2_p;
    ff_dim_t softmax_dim = ff_dim_t{1_n};
    ff_dim_t partition_dim = ff_dim_t{0_n};
    std::string softmax_match = "softmax_match";

    Substitution sub =
        create_partition_softmax_combine(softmax_dim, partition_dim, degree);

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                10_p,
                10_p,
            },
        },
        DataType::FLOAT,
    };

    SoftmaxAttrs softmax_attrs = SoftmaxAttrs{
        /*softmax_dim=*/softmax_dim,
    };

    SubParallelComputationGraph original_pcg = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      parallel_tensor_guid_t t_input = add_input_layer(pcg, input_shape);

      parallel_tensor_guid_t t_softmax = add_single_output_layer(
          pcg,
          make_layer_attrs(softmax_attrs, softmax_match),
          {{TensorSlotName::INPUT, t_input}},
          {});

      return sub_pcg_from_full_pcg(pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t match_layer =
          get_parallel_layer_by_name(original_pcg, softmax_match);
      open_parallel_tensor_guid_t match_layer_input =
          get_layer_inputs(original_pcg, match_layer).at(TensorSlotName::INPUT);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, match_layer},
          },
          std::map<PatternInput, open_parallel_tensor_guid_t>{{
              PatternInput{KwargDataflowGraphInput{0}},
              match_layer_input,
          }},
      };
    }();

    SubParallelComputationGraph result =
        apply_substitution(original_pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      parallel_tensor_guid_t t_partitioned_input = add_partition_layer(
          pcg, partition_dim, degree, add_input_layer(pcg, input_shape));

      parallel_tensor_guid_t t_partitioned_softmax = add_single_output_layer(
          pcg,
          make_layer_attrs(softmax_attrs),
          {{TensorSlotName::INPUT, t_partitioned_input}},
          {});

      parallel_tensor_guid_t t_combine =
          add_combine_layer(pcg, partition_dim, degree, t_partitioned_softmax);

      return sub_pcg_from_full_pcg(pcg);
    }();

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }

  TEST_CASE("create_partition_add_combine") {
    positive_int degree = 2_p;
    ff_dim_t parallel_dim = ff_dim_t{1_n};
    std::string add_match = "add_match";

    Substitution sub = create_partition_add_combine(parallel_dim, degree);

    TensorShape lhs_shape = TensorShape{
        TensorDims{
            FFOrdered{
                10_p,
                15_p,
            },
        },
        DataType::FLOAT,
    };

    TensorShape rhs_shape = lhs_shape;

    ElementBinaryAttrs add_attrs = ElementBinaryAttrs{
        ElementBinaryOp::ADD,
        DataType::FLOAT,
        false,
        false,
    };

    SubParallelComputationGraph original_pcg = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      parallel_tensor_guid_t t_lhs = add_input_layer(pcg, lhs_shape);
      parallel_tensor_guid_t t_rhs = add_input_layer(pcg, rhs_shape);

      parallel_tensor_guid_t t_add =
          add_single_output_layer(pcg,
                                  make_layer_attrs(add_attrs, add_match),
                                  {
                                      {TensorSlotName::LHS_INPUT, t_lhs},
                                      {TensorSlotName::RHS_INPUT, t_rhs},
                                  },
                                  {});

      return sub_pcg_from_full_pcg(pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t match_layer =
          get_parallel_layer_by_name(original_pcg, add_match);
      open_parallel_tensor_guid_t add_match_layer_lhs =
          get_layer_inputs(original_pcg, match_layer)
              .at(TensorSlotName::LHS_INPUT);
      open_parallel_tensor_guid_t add_match_layer_rhs =
          get_layer_inputs(original_pcg, match_layer)
              .at(TensorSlotName::RHS_INPUT);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, match_layer},
          },
          std::map<PatternInput, open_parallel_tensor_guid_t>{
              {
                  PatternInput{KwargDataflowGraphInput{0}},
                  add_match_layer_lhs,
              },
              {
                  PatternInput{KwargDataflowGraphInput{2}},
                  add_match_layer_rhs,
              }},
      };
    }();

    SubParallelComputationGraph result =
        apply_substitution(original_pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      parallel_tensor_guid_t t_lhs = add_partition_layer(
          pcg, parallel_dim, degree, add_input_layer(pcg, lhs_shape));
      parallel_tensor_guid_t t_rhs = add_partition_layer(
          pcg, parallel_dim, degree, add_input_layer(pcg, rhs_shape));

      parallel_tensor_guid_t t_partitioned_add =
          add_single_output_layer(pcg,
                                  make_layer_attrs(add_attrs, add_match),
                                  {
                                      {TensorSlotName::LHS_INPUT, t_lhs},
                                      {TensorSlotName::RHS_INPUT, t_rhs},
                                  },
                                  {});

      parallel_tensor_guid_t t_combine =
          add_combine_layer(pcg, parallel_dim, degree, t_partitioned_add);

      return sub_pcg_from_full_pcg(pcg);
    }();

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }

  TEST_CASE("create_partition_relu_combine") {
    positive_int degree = 2_p;
    ff_dim_t parallel_dim = ff_dim_t{1_n};
    std::string relu_match = "relu_match";

    Substitution sub = create_partition_relu_combine(parallel_dim, degree);

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                10_p,
                10_p,
            },
        },
        DataType::FLOAT,
    };

    ElementUnaryAttrs relu_attrs = ElementUnaryAttrs{
        ElementUnaryOp::RELU,
        std::nullopt,
    };

    SubParallelComputationGraph original_pcg = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      parallel_tensor_guid_t t_input = add_input_layer(pcg, input_shape);

      parallel_tensor_guid_t t_relu =
          add_single_output_layer(pcg,
                                  make_layer_attrs(relu_attrs, relu_match),
                                  {{TensorSlotName::INPUT, t_input}},
                                  {});

      return sub_pcg_from_full_pcg(pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t match_layer =
          get_parallel_layer_by_name(original_pcg, relu_match);
      open_parallel_tensor_guid_t match_layer_input =
          get_layer_inputs(original_pcg, match_layer).at(TensorSlotName::INPUT);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, match_layer},
          },
          std::map<PatternInput, open_parallel_tensor_guid_t>{{
              PatternInput{KwargDataflowGraphInput{0}},
              match_layer_input,
          }},
      };
    }();

    SubParallelComputationGraph result =
        apply_substitution(original_pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      parallel_tensor_guid_t t_input = add_partition_layer(
          pcg, parallel_dim, degree, add_input_layer(pcg, input_shape));

      parallel_tensor_guid_t t_relu =
          add_single_output_layer(pcg,
                                  make_layer_attrs(relu_attrs),
                                  {{TensorSlotName::INPUT, t_input}},
                                  {});

      parallel_tensor_guid_t t_combine =
          add_combine_layer(pcg, parallel_dim, degree, t_relu);

      return sub_pcg_from_full_pcg(pcg);
    }();

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }

  TEST_CASE("create_fuse_linear_activation") {
    Substitution sub = create_fuse_linear_activation(Activation::SIGMOID);

    std::string mm_match = "mm_match";
    std::string relu_match = "relu_match";

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                4_p,
                10_p,
            },
        },
        DataType::FLOAT,
    };

    SubParallelComputationGraph pcg = [&] {
      ParallelComputationGraphBuilder b;
      parallel_tensor_guid_t t = b.create_input_tensor(input_shape);
      t = b.dense(t,
                  /*outDim=*/4_p,
                  /*activation=*/std::nullopt,
                  /*use_bias=*/false,
                  /*data_type=*/DataType::FLOAT,
                  /*kernel_initializer=*/std::nullopt,
                  /*bias_initializer=*/std::nullopt,
                  /*name=*/mm_match);
      t = b.relu(t,
                 /*name=*/relu_match);

      return sub_pcg_from_full_pcg(b.pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t mm_match_layer =
          get_parallel_layer_by_name(pcg, mm_match);
      parallel_layer_guid_t relu_match_layer =
          get_parallel_layer_by_name(pcg, relu_match);
      open_parallel_tensor_guid_t mm_match_layer_input_activations =
          get_layer_inputs(pcg, mm_match_layer).at(TensorSlotName::INPUT);
      open_parallel_tensor_guid_t mm_match_layer_input_weights =
          get_layer_inputs(pcg, mm_match_layer).at(TensorSlotName::WEIGHT);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, mm_match_layer},
              {PatternNode{Node{1}}, relu_match_layer},
          },
          std::map<PatternInput, open_parallel_tensor_guid_t>{
              {
                  PatternInput{KwargDataflowGraphInput{0}},
                  mm_match_layer_input_activations,
              },
              {
                  PatternInput{KwargDataflowGraphInput{2}},
                  mm_match_layer_input_weights,
              }},
      };
    }();

    SubParallelComputationGraph result = apply_substitution(pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraphBuilder b;
      parallel_tensor_guid_t t = b.create_input_tensor(input_shape);
      t = b.dense(t,
                  /*outDim=*/4_p,
                  /*activation=*/Activation::SIGMOID,
                  /*use_bias=*/false,
                  /*data_type=*/DataType::FLOAT,
                  /*kernel_initializer=*/std::nullopt,
                  /*bias_initializer=*/std::nullopt,
                  /*name=*/std::nullopt);

      return sub_pcg_from_full_pcg(b.pcg);
    }();

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }

  TEST_CASE("create_fuse_batch_norm_activation" * doctest::should_fail(true)) {
    Substitution sub = create_fuse_batch_norm_activation(Activation::SILU);

    std::string bn_match = "bn_match";
    std::string silu_match = "silu_match";

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                4_p,
                10_p,
            },
        },
        DataType::FLOAT,
    };

    SubParallelComputationGraph pcg = [&] {
      ParallelComputationGraphBuilder b;
      parallel_tensor_guid_t t = b.create_input_tensor(input_shape);
      t = b.batch_norm(t,
                       /*affine=*/true,
                       /*activation=*/std::nullopt,
                       /*eps=*/0.5,
                       /*momentum=*/std::nullopt,
                       /*name=*/bn_match);
      t = b.silu(t, /*name=*/silu_match);

      return sub_pcg_from_full_pcg(b.pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t bn_match_layer =
          get_parallel_layer_by_name(pcg, bn_match);
      parallel_layer_guid_t silu_match_layer =
          get_parallel_layer_by_name(pcg, silu_match);
      open_parallel_tensor_guid_t bn_match_layer_input_activations =
          get_layer_inputs(pcg, bn_match_layer).at(TensorSlotName::INPUT);
      open_parallel_tensor_guid_t bn_match_layer_input_gamma_weights =
          get_layer_inputs(pcg, bn_match_layer).at(TensorSlotName::GAMMA);
      open_parallel_tensor_guid_t bn_match_layer_input_beta_weights =
          get_layer_inputs(pcg, bn_match_layer).at(TensorSlotName::BETA);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, bn_match_layer},
              {PatternNode{Node{1}}, silu_match_layer},
          },
          std::map<PatternInput, open_parallel_tensor_guid_t>{
              {
                  // TODO(@lockshaw): change input ids to strings so that the id doesn't have to be guessed like below
                  PatternInput{KwargDataflowGraphInput{0}},
                  bn_match_layer_input_activations,
              },
              {
                  PatternInput{KwargDataflowGraphInput{2}},
                  bn_match_layer_input_gamma_weights,
              },
              {
                  PatternInput{KwargDataflowGraphInput{4}},
                  bn_match_layer_input_beta_weights,
              },
          },
      };
    }();

    SubParallelComputationGraph result = apply_substitution(pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraphBuilder b;
      parallel_tensor_guid_t t = b.create_input_tensor(input_shape);
      t = b.batch_norm(t,
                       /*affine=*/true,
                       /*activation=*/Activation::SILU,
                       /*eps=*/0.5,
                       /*momentum=*/std::nullopt,
                       /*name=*/std::nullopt);

      return sub_pcg_from_full_pcg(b.pcg);
    }();

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }
}

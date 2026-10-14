#include "substitutions/apply_substitution/evaluate_substitution_output.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "substitutions/open_parallel_tensor_guid_t.h"
#include "substitutions/operator_pattern/operator_attribute_constraint.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"
#include "utils/containers/get_only.h"
#include "utils/containers/require_only_key.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("evaluate_substitution_output") {
    // Currently Substitution creation is very verbose.
    // This is being addressed in
    // https://github.com/flexflow/FlexFlow/issues/1473.
    auto pattern_g = LabelledOpenKwargDataflowGraph<OperatorAttributePattern,
                                                    TensorAttributePattern,
                                                    int,
                                                    TensorSlotName>::
        create<
            UnorderedSetLabelledOpenKwargDataflowGraph<OperatorAttributePattern,
                                                       TensorAttributePattern,
                                                       int,
                                                       TensorSlotName>>();

    PatternInput pattern_i_activation = PatternInput{
        pattern_g.add_input(0, tensor_attribute_pattern_match_all())};
    PatternInput pattern_i_weights = PatternInput{
        pattern_g.add_input(1, tensor_attribute_pattern_match_all())};

    OperatorAttributePattern mm_pattern = OperatorAttributePattern{{
        op_type_equals_constraint(OperatorType::LINEAR),
        op_attr_key_equals(
            OperatorAttributeKey::ACTIVATION,
            OperatorAttributeValue{std::optional<Activation>{std::nullopt}}),
    }};
    KwargNodeAddedResult mm_added = pattern_g.add_node(
        /*node_label=*/mm_pattern,
        /*inputs=*/
        {
            {
                TensorSlotName::INPUT,
                OpenKwargDataflowValue<int, TensorSlotName>{
                    pattern_i_activation.raw_dataflow_graph_input,
                },
            },
            {
                TensorSlotName::WEIGHT,
                OpenKwargDataflowValue<int, TensorSlotName>{
                    pattern_i_weights.raw_dataflow_graph_input,
                },
            },
        },
        /*output_labels=*/
        {
            {
                TensorSlotName::OUTPUT,
                tensor_attribute_pattern_match_all(),
            },
        });
    PatternNode pattern_mm_node = PatternNode{mm_added.node};
    KwargDataflowOutput<TensorSlotName> mm_output =
        require_only_key(mm_added.outputs, TensorSlotName::OUTPUT);

    OperatorAttributePattern relu_pattern = OperatorAttributePattern{{
        op_type_equals_constraint(OperatorType::RELU),
    }};
    KwargNodeAddedResult relu_added = pattern_g.add_node(
        /*node_label=*/relu_pattern,
        /*inputs=*/
        {
            {
                TensorSlotName::INPUT,
                OpenKwargDataflowValue<int, TensorSlotName>{mm_output},
            },
        },
        /*output_labels=*/
        {
            {
                TensorSlotName::OUTPUT,
                tensor_attribute_pattern_match_all(),
            },
        });
    PatternNode pattern_relu_node = PatternNode{relu_added.node};
    KwargDataflowOutput<TensorSlotName> relu_output =
        require_only_key(relu_added.outputs, TensorSlotName::OUTPUT);

    LabelledOpenKwargDataflowGraph<OutputOperatorAttrsAssignment,
                                   std::monostate,
                                   int,
                                   TensorSlotName>
        output_g = LabelledOpenKwargDataflowGraph<OutputOperatorAttrsAssignment,
                                                  std::monostate,
                                                  int,
                                                  TensorSlotName>::
            create<UnorderedSetLabelledOpenKwargDataflowGraph<
                OutputOperatorAttrsAssignment,
                std::monostate,
                int,
                TensorSlotName>>();

    OutputGraphExprInput output_i_activation =
        OutputGraphExprInput{output_g.add_input(0, std::monostate{})};
    OutputGraphExprInput output_i_weights =
        OutputGraphExprInput{output_g.add_input(1, std::monostate{})};

    OutputOperatorAttrsAssignment fused_mm_relu_attrs_assignment =
        OutputOperatorAttrsAssignment{
            std::nullopt,
            {
                set_attr_to_constant(
                    OperatorAttributeKey::OP_TYPE,
                    OperatorAttributeValue{OperatorType::LINEAR}),
                copy_attr_from_pattern_node(OperatorAttributeKey::OUT_CHANNELS,
                                            pattern_mm_node),
                copy_attr_from_pattern_node(OperatorAttributeKey::USE_BIAS,
                                            pattern_mm_node),
                copy_attr_from_pattern_node(OperatorAttributeKey::DATA_TYPE,
                                            pattern_mm_node),
                set_attr_to_constant(OperatorAttributeKey::ACTIVATION,
                                     OperatorAttributeValue{Activation::RELU}),
                copy_attr_from_pattern_node(OperatorAttributeKey::REGULARIZER,
                                            pattern_mm_node),
            }};
    KwargNodeAddedResult fused_mm_relu_added = output_g.add_node(
        /*node_label=*/fused_mm_relu_attrs_assignment,
        /*inputs=*/
        {
            {
                TensorSlotName::INPUT,
                OpenKwargDataflowValue<int, TensorSlotName>{
                    output_i_activation.raw_dataflow_graph_input,
                },
            },
            {
                TensorSlotName::WEIGHT,
                OpenKwargDataflowValue<int, TensorSlotName>{
                    output_i_weights.raw_dataflow_graph_input,
                },
            },
        },
        /*output_labels=*/
        {
            {
                TensorSlotName::OUTPUT,
                std::monostate{},
            },
        });
    OutputGraphExprNode fused_mm_relu_node =
        OutputGraphExprNode{fused_mm_relu_added.node};
    KwargDataflowOutput<TensorSlotName> fused_mm_relu_output =
        require_only_key(fused_mm_relu_added.outputs, TensorSlotName::OUTPUT);

    Substitution sub = Substitution{
        PCGPattern{pattern_g},
        OutputGraphExpr{output_g},
        bidict<PatternInput, OutputGraphExprInput>{
            {
                pattern_i_activation,
                output_i_activation,
            },
            {
                pattern_i_weights,
                output_i_weights,
            },
        },
        bidict<PatternNodeOutput, OutputGraphExprNodeOutput>{
            {
                PatternNodeOutput{relu_output},
                OutputGraphExprNodeOutput{fused_mm_relu_output},
            },
        },
    };

    positive_int in_channels = 24_p;
    positive_int batch_size = 4_p;
    int_ge_two batch_degree = 2_ge2;
    std::string mm_match = "mm_match";
    std::string relu_match = "relu_match";

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                batch_size,
                in_channels,
            },
        },
        DataType::FLOAT,
    };

    SubParallelComputationGraph pcg = [&] {
      ParallelComputationGraphBuilder b;

      parallel_tensor_guid_t t = b.create_input_tensor(input_shape);
      t = b.parallel_partition(t, ff_dim_t{0_n}, batch_degree);
      t = b.dense(t,
                  /*outDim=*/16_p,
                  /*activation=*/std::nullopt);
      t = b.gelu(t);
      t = b.dense(t,
                  /*outDim=*/12_p,
                  /*activation=*/std::nullopt,
                  /*use_bias=*/false,
                  /*data_type=*/DataType::FLOAT,
                  /*kernel_initializer=*/std::nullopt,
                  /*bias_initializer=*/std::nullopt,
                  /*name=*/mm_match);
      t = b.relu(t,
                 /*name=*/relu_match);
      t = b.dense(t,
                  /*outDim=*/8_p,
                  /*activation=*/Activation::RELU);

      return sub_pcg_from_full_pcg(b.pcg);
    }();

    parallel_layer_guid_t mm_match_layer =
        get_parallel_layer_by_name(pcg, mm_match);
    parallel_layer_guid_t relu_match_layer =
        get_parallel_layer_by_name(pcg, relu_match);
    open_parallel_tensor_guid_t mm_match_layer_input_activations =
        get_layer_inputs(pcg, mm_match_layer).at(TensorSlotName::INPUT);
    open_parallel_tensor_guid_t mm_match_layer_input_weights =
        get_layer_inputs(pcg, mm_match_layer).at(TensorSlotName::WEIGHT);

    PCGPatternMatch match = PCGPatternMatch{
        bidict<PatternNode, parallel_layer_guid_t>{
            {pattern_mm_node, mm_match_layer},
            {pattern_relu_node, relu_match_layer},
        },
        std::map<PatternInput, open_parallel_tensor_guid_t>{
            {
                PatternInput{pattern_i_activation},
                mm_match_layer_input_activations,
            },
            {
                PatternInput{pattern_i_weights},
                mm_match_layer_input_weights,
            }},
    };

    SUBCASE("evaluate_substitution_output") {
      std::pair<SubParallelComputationGraph, OutputExprToResultSubPCGMapping>
          result = evaluate_substitution_output(pcg, sub, match);

      SubParallelComputationGraph result_graph = result.first;
      bidict<parallel_layer_guid_t, OutputGraphExprNode> result_node_map =
          result.second.node_mapping;
      bidict<input_parallel_tensor_guid_t, OutputGraphExprInput>
          result_input_map = result.second.input_mapping;

      LinearAttrs correct_result_fused_mm_relu_attrs = LinearAttrs{
          /*out_channels=*/12_p,
          /*use_bias=*/false,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/Activation::RELU,
          /*regularizer=*/std::nullopt,
      };

      ParallelTensorAttrs correct_result_i_activation_attrs =
          get_parallel_tensor_attrs(pcg, mm_match_layer_input_activations);
      ParallelTensorAttrs correct_result_i_weights_attrs =
          get_parallel_tensor_attrs(pcg, mm_match_layer_input_weights);
      ParallelTensorAttrs correct_result_fused_mm_relu_output_attrs =
          get_parallel_tensor_attrs(
              pcg,
              open_parallel_tensor_guid_from_closed(
                  require_only_key(get_outgoing_tensors(pcg, relu_match_layer),
                                   TensorSlotName::OUTPUT)));

      parallel_layer_guid_t result_fused_mm_relu_node =
          result_node_map.at_r(fused_mm_relu_node);
      parallel_tensor_guid_t result_fused_mm_relu_output = require_only_key(
          get_outgoing_tensors(result_graph, result_fused_mm_relu_node),
          TensorSlotName::OUTPUT);
      input_parallel_tensor_guid_t result_i_activation =
          result_input_map.at_r(output_i_activation);
      input_parallel_tensor_guid_t result_i_weights =
          result_input_map.at_r(output_i_weights);

      SubParallelComputationGraphData correct_graph_data =
          SubParallelComputationGraphData{
              std::map<parallel_layer_guid_t, ParallelLayerAttrs>{{
                  result_fused_mm_relu_node,
                  ParallelLayerAttrs{
                      PCGOperatorAttrs{correct_result_fused_mm_relu_attrs},
                      /*name=*/std::nullopt,
                  },
              }},
              std::set<SubParallelComputationGraphEdge>{
                  SubParallelComputationGraphEdge{
                      OpenKwargDataflowEdge<int, TensorSlotName>{
                          KwargDataflowInputEdge{
                              result_i_activation.raw_dataflow_graph_input,
                              KwargDataflowInput<TensorSlotName>{
                                  result_fused_mm_relu_node.raw_graph_node,
                                  TensorSlotName::INPUT,
                              },
                          },
                      },
                  },
                  SubParallelComputationGraphEdge{
                      OpenKwargDataflowEdge<int, TensorSlotName>{
                          KwargDataflowInputEdge<int, TensorSlotName>{
                              result_i_weights.raw_dataflow_graph_input,
                              KwargDataflowInput<TensorSlotName>{
                                  result_fused_mm_relu_node.raw_graph_node,
                                  TensorSlotName::WEIGHT,
                              },
                          },
                      },
                  },
              },
              std::set<input_parallel_tensor_guid_t>{
                  result_i_activation,
                  result_i_weights,
              },
              std::map<open_parallel_tensor_guid_t, ParallelTensorAttrs>{
                  {
                      open_parallel_tensor_guid_from_input(result_i_activation),
                      correct_result_i_activation_attrs,
                  },
                  {
                      open_parallel_tensor_guid_from_input(result_i_weights),
                      correct_result_i_weights_attrs,
                  },
                  {
                      open_parallel_tensor_guid_from_closed(
                          result_fused_mm_relu_output),
                      correct_result_fused_mm_relu_output_attrs,
                  }}};

      SubParallelComputationGraphData result_graph_data =
          get_sub_pcg_data(result_graph);

      CHECK(result_graph_data == correct_graph_data);
    }
  }
}

#include "substitutions/substitution_internal/evaluate_substitution_output.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "substitutions/open_parallel_tensor_guid_t.h"
#include "substitutions/operator_pattern/operator_attribute_constraint.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"
#include "utils/containers/get_only.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("evaluate_substitution_output") {
    // Currently Substitution creation is very verbose.
    // This is being addressed in
    // https://github.com/flexflow/FlexFlow/issues/1473.
    auto pattern_g = LabelledOpenDataflowGraph<OperatorAttributePattern,
                                               TensorAttributePattern>::
        create<UnorderedSetLabelledOpenDataflowGraph<OperatorAttributePattern,
                                                     TensorAttributePattern>>();

    PatternInput pattern_i_activation =
        PatternInput{pattern_g.add_input(tensor_attribute_pattern_match_all())};
    PatternInput pattern_i_weights =
        PatternInput{pattern_g.add_input(tensor_attribute_pattern_match_all())};

    OperatorAttributePattern mm_pattern = OperatorAttributePattern{{
        op_type_equals_constraint(OperatorType::LINEAR),
        op_attr_key_equals(
            OperatorAttributeKey::ACTIVATION,
            OperatorAttributeValue{std::optional<Activation>{std::nullopt}}),
    }};
    NodeAddedResult mm_added = pattern_g.add_node(
        mm_pattern,
        {OpenDataflowValue{pattern_i_activation.raw_dataflow_graph_input},
         OpenDataflowValue{pattern_i_weights.raw_dataflow_graph_input}},
        {tensor_attribute_pattern_match_all()});
    PatternNode pattern_mm_node = PatternNode{mm_added.node};
    DataflowOutput mm_output = get_only(mm_added.outputs);

    OperatorAttributePattern relu_pattern = OperatorAttributePattern{{
        op_type_equals_constraint(OperatorType::RELU),
    }};
    NodeAddedResult relu_added =
        pattern_g.add_node(relu_pattern,
                           {OpenDataflowValue{mm_output}},
                           {tensor_attribute_pattern_match_all()});
    PatternNode pattern_relu_node = PatternNode{relu_added.node};
    DataflowOutput relu_output = get_only(relu_added.outputs);

    LabelledOpenDataflowGraph<OutputOperatorAttrsAssignment, std::monostate>
        output_g = LabelledOpenDataflowGraph<OutputOperatorAttrsAssignment,
                                             std::monostate>::
            create<UnorderedSetLabelledOpenDataflowGraph<
                OutputOperatorAttrsAssignment,
                std::monostate>>();

    OutputGraphExprInput output_i_activation =
        OutputGraphExprInput{output_g.add_input({})};
    OutputGraphExprInput output_i_weights =
        OutputGraphExprInput{output_g.add_input({})};

    OutputOperatorAttrsAssignment fused_mm_relu_attrs_assignment =
        OutputOperatorAttrsAssignment{{
            set_attr_to_constant(OperatorAttributeKey::OP_TYPE,
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
    NodeAddedResult fused_mm_relu_added = output_g.add_node(
        fused_mm_relu_attrs_assignment,
        {OpenDataflowValue{output_i_activation.raw_dataflow_graph_input},
         OpenDataflowValue{output_i_weights.raw_dataflow_graph_input}},
        {{}});
    OutputGraphExprNode fused_mm_relu_node =
        OutputGraphExprNode{fused_mm_relu_added.node};
    DataflowOutput fused_mm_relu_output = get_only(fused_mm_relu_added.outputs);

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

    int in_channels = 24;
    int batch_size = 4;
    int batch_degree = 2;
    std::string mm_match = "mm_match";
    std::string relu_match = "relu_match";

    SubParallelComputationGraph pcg = [&] {
      ParallelComputationGraphBuilder b;
      parallel_tensor_guid_t t = b.create_input_tensor(ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{size_t_from_int(batch_size), batch_degree},
                  ShardParallelDim{size_t_from_int(in_channels), 1},
              },
              ReplicaParallelDimSet{
                  SumDegree{1},
                  DiscardCopyDegree{1},
              },
          },
          DataType::FLOAT,
      });
      t = b.dense(t,
                  /*outDim=*/16,
                  /*activation=*/std::nullopt);
      t = b.gelu(t);
      t = b.dense(t,
                  /*outDim=*/12,
                  /*activation=*/std::nullopt,
                  /*use_bias=*/false,
                  /*data_type=*/DataType::FLOAT,
                  /*kernel_initializer=*/std::nullopt,
                  /*bias_initializer=*/std::nullopt,
                  /*name=*/mm_match);
      t = b.relu(t,
                 /*name=*/relu_match);
      t = b.dense(t,
                  /*outDim=*/8,
                  /*activation=*/Activation::RELU);

      return sub_pcg_from_full_pcg(b.pcg);
    }();

    parallel_layer_guid_t mm_match_layer =
        get_parallel_layer_by_name(pcg, mm_match);
    parallel_layer_guid_t relu_match_layer =
        get_parallel_layer_by_name(pcg, relu_match);
    open_parallel_tensor_guid_t mm_match_layer_input_activations =
        get_layer_inputs(pcg, mm_match_layer).at(0);
    open_parallel_tensor_guid_t mm_match_layer_input_weights =
        get_layer_inputs(pcg, mm_match_layer).at(1);

    PCGPatternMatch match = PCGPatternMatch{
        bidict<PatternNode, parallel_layer_guid_t>{
            {pattern_mm_node, mm_match_layer},
            {pattern_relu_node, relu_match_layer},
        },
        std::unordered_map<PatternInput, open_parallel_tensor_guid_t>{
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
          12,
          /*use_bias=*/false,
          DataType::FLOAT,
          Activation::RELU,
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
                  get_only(get_layer_outputs(pcg, relu_match_layer))));

      parallel_layer_guid_t result_fused_mm_relu_node =
          result_node_map.at_r(fused_mm_relu_node);
      parallel_tensor_guid_t result_fused_mm_relu_output =
          get_only(get_layer_outputs(result_graph, result_fused_mm_relu_node));
      input_parallel_tensor_guid_t result_i_activation =
          result_input_map.at_r(output_i_activation);
      input_parallel_tensor_guid_t result_i_weights =
          result_input_map.at_r(output_i_weights);

      SubParallelComputationGraphData correct_graph_data =
          SubParallelComputationGraphData{
              std::unordered_map<parallel_layer_guid_t, ParallelLayerAttrs>{{
                  result_fused_mm_relu_node,
                  ParallelLayerAttrs{
                      PCGOperatorAttrs{correct_result_fused_mm_relu_attrs},
                      /*name=*/std::nullopt,
                  },
              }},
              std::unordered_set<SubParallelComputationGraphEdge>{
                  SubParallelComputationGraphEdge{
                      OpenDataflowEdge{
                          DataflowInputEdge{
                              result_i_activation.raw_dataflow_graph_input,
                              DataflowInput{
                                  result_fused_mm_relu_node.raw_graph_node,
                                  0,
                              },
                          },
                      },
                  },
                  SubParallelComputationGraphEdge{
                      OpenDataflowEdge{
                          DataflowInputEdge{
                              result_i_weights.raw_dataflow_graph_input,
                              DataflowInput{
                                  result_fused_mm_relu_node.raw_graph_node,
                                  1,
                              },
                          },
                      },
                  },
              },
              std::unordered_set<input_parallel_tensor_guid_t>{
                  result_i_activation,
                  result_i_weights,
              },
              std::unordered_map<open_parallel_tensor_guid_t,
                                 ParallelTensorAttrs>{
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

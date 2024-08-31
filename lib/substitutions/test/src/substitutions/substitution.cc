#include "substitutions/substitution.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "substitutions/open_parallel_tensor_guid_t.h"
#include "substitutions/operator_pattern/operator_attribute_constraint.h"
#include "substitutions/output_graph/output_graph_expr_node.dtg.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/pcg_pattern_builder.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"
#include "utils/containers/get_only.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/get_graph_data.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  // TEST_CASE("is_valid_substitution") {
  //   FAIL("TODO");
  // }

  TEST_CASE("evaluate_substitution_output(SubParallelComputationGraph, "
            "Substituion, PCGPatternMatch)") {
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

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t mm_match_layer =
          get_parallel_layer_by_name(pcg, mm_match);
      parallel_layer_guid_t relu_match_layer =
          get_parallel_layer_by_name(pcg, relu_match);
      open_parallel_tensor_guid_t mm_match_layer_input_activations =
          get_layer_inputs(pcg, mm_match_layer).at(0);
      open_parallel_tensor_guid_t mm_match_layer_input_weights =
          get_layer_inputs(pcg, mm_match_layer).at(1);

      return PCGPatternMatch{
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
    }();

    SubParallelComputationGraph result = apply_substitution(pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
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
                  /*activation=*/Activation::RELU,
                  /*use_bias=*/false,
                  /*data_type=*/DataType::FLOAT,
                  /*kernel_initializer=*/std::nullopt,
                  /*bias_initializer=*/std::nullopt,
                  /*name=*/std::nullopt);
      t = b.dense(t,
                  /*outDim=*/8,
                  /*activation=*/Activation::RELU);

      return sub_pcg_from_full_pcg(b.pcg);
    }();

    // since the new nodes produced by the substitution have new ids, it's
    // easier/more correct to check that the graphs are isomorphic rather than
    // checking their exact graph data
    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }
}

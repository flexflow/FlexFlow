#include "substitutions/substitution_builder.h"
#include "substitutions/operator_pattern/operator_attribute_constraint.h"
#include "substitutions/output_graph/output_graph_expr_node.dtg.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/substitution.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"
#include "utils/containers/get_only.h"
#include "utils/containers/require_only_key.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("SubstitutionBuilder") {
    OperatorAttributePattern relu_pattern = OperatorAttributePattern{{
        op_type_equals_constraint(OperatorType::RELU),
    }};

    OperatorAttributePattern mm_pattern = OperatorAttributePattern{{
        op_type_equals_constraint(OperatorType::LINEAR),
        op_attr_key_equals(
            OperatorAttributeKey::ACTIVATION,
            OperatorAttributeValue{std::optional<Activation>{std::nullopt}}),
    }};

    std::map<OperatorAttributeKey, OutputOperatorAttributeExpr>
        fused_mm_relu_attr_assignments = {
            set_attr_to_constant(OperatorAttributeKey::ACTIVATION,
                                 OperatorAttributeValue{Activation::RELU}),
        };

    Substitution correct = [&] {
      auto pattern_g = LabelledOpenKwargDataflowGraph<OperatorAttributePattern,
                                                      TensorAttributePattern,
                                                      int,
                                                      TensorSlotName>::
          create<UnorderedSetLabelledOpenKwargDataflowGraph<
              OperatorAttributePattern,
              TensorAttributePattern,
              int,
              TensorSlotName>>();

      PatternInput pattern_i_activation = PatternInput{
          pattern_g.add_input(0, tensor_attribute_pattern_match_all()),
      };
      PatternInput pattern_i_weights = PatternInput{
          pattern_g.add_input(1, tensor_attribute_pattern_match_all()),
      };

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
          output_g =
              LabelledOpenKwargDataflowGraph<OutputOperatorAttrsAssignment,
                                             std::monostate,
                                             int,
                                             TensorSlotName>::
                  create<UnorderedSetLabelledOpenKwargDataflowGraph<
                      OutputOperatorAttrsAssignment,
                      std::monostate,
                      int,
                      TensorSlotName>>();

      OutputGraphExprInput output_i_activation =
          OutputGraphExprInput{output_g.add_input(0, {})};
      OutputGraphExprInput output_i_weights =
          OutputGraphExprInput{output_g.add_input(1, {})};

      OutputOperatorAttrsAssignment fused_mm_relu_attrs_assignment =
          OutputOperatorAttrsAssignment{
              pattern_mm_node,
              fused_mm_relu_attr_assignments,
          };
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
          {{
              TensorSlotName::OUTPUT,
              std::monostate{},
          }});
      OutputGraphExprNode fused_mm_relu_node =
          OutputGraphExprNode{fused_mm_relu_added.node};
      KwargDataflowOutput<TensorSlotName> fused_mm_relu_output =
          require_only_key(fused_mm_relu_added.outputs, TensorSlotName::OUTPUT);

      return Substitution{
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
    }();

    Substitution result = [&] {
      //! [SubstitutionBuilder example]
      SubstitutionBuilder b;

      auto [p_input, o_input] =
          b.add_input(tensor_attribute_pattern_match_all());
      auto [p_weight, o_weight] =
          b.add_input(tensor_attribute_pattern_match_all());

      PatternValue p_mm_output =
          require_only_key(b.add_pattern_node(
                               /*node_pattern=*/mm_pattern,
                               /*inputs=*/
                               {
                                   {
                                       TensorSlotName::INPUT,
                                       p_input,
                                   },
                                   {
                                       TensorSlotName::WEIGHT,
                                       p_weight,
                                   },
                               },
                               /*output_patterns=*/
                               {
                                   {
                                       TensorSlotName::OUTPUT,
                                       tensor_attribute_pattern_match_all(),
                                   },
                               },
                               /*name=*/"mm"),
                           TensorSlotName::OUTPUT);

      PatternValue p_relu_output =
          require_only_key(b.add_pattern_node(
                               /*node_pattern=*/relu_pattern,
                               /*inputs=*/
                               {
                                   {
                                       TensorSlotName::INPUT,
                                       p_mm_output,
                                   },
                               },
                               /*output_patterns=*/
                               {
                                   {
                                       TensorSlotName::OUTPUT,
                                       tensor_attribute_pattern_match_all(),
                                   },
                               },
                               /*name=*/"relu"),
                           TensorSlotName::OUTPUT);

      OutputOperatorAttrsAssignment fused_mm_relu_attrs_assignment =
          OutputOperatorAttrsAssignment{
              b.pattern_node_named("mm"),
              fused_mm_relu_attr_assignments,
          };
      OutputGraphExprValue o_fused_output =
          require_only_key(b.add_output_graph_node(
                               /*node_expr=*/fused_mm_relu_attrs_assignment,
                               /*inputs=*/
                               {
                                   {
                                       TensorSlotName::INPUT,
                                       o_input,
                                   },
                                   {
                                       TensorSlotName::WEIGHT,
                                       o_weight,
                                   },
                               },
                               /*output_slots=*/
                               {
                                   TensorSlotName::OUTPUT,
                               }),
                           TensorSlotName::OUTPUT);

      b.equate_outputs(p_relu_output, o_fused_output);

      return b.get_substitution();
      //! [SubstitutionBuilder example]
    }();

    CHECK(is_isomorphic_to(result, correct));
  }
}

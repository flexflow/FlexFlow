#include "substitutions/substitution_builder.h"
#include "substitutions/output_graph/output_graph_expr_value.h"
#include "substitutions/substitution.h"
#include "substitutions/unlabelled/pattern_value.h"
#include "utils/containers/repeat_element.h"
#include "utils/exception.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/overload.h"

namespace FlexFlow {

SubstitutionBuilder::SubstitutionBuilder()
    : pattern_g(LabelledOpenKwargDataflowGraph<OperatorAttributePattern,
                                               TensorAttributePattern,
                                               int,
                                               TensorSlotName>::
                    create<UnorderedSetLabelledOpenKwargDataflowGraph<
                        OperatorAttributePattern,
                        TensorAttributePattern,
                        int,
                        TensorSlotName>>()),
      output_g(LabelledOpenKwargDataflowGraph<OutputOperatorAttrsAssignment,
                                              std::monostate,
                                              int,
                                              TensorSlotName>::
                   create<UnorderedSetLabelledOpenKwargDataflowGraph<
                       OutputOperatorAttrsAssignment,
                       std::monostate,
                       int,
                       TensorSlotName>>()),
      next_graph_input_id{0} {}

std::pair<PatternValue, OutputGraphExprValue> SubstitutionBuilder::add_input(
    TensorAttributePattern const &input_tensor_pattern,
    std::optional<std::string> const &name) {
  PatternInput pattern_input = PatternInput{
      this->pattern_g.add_input(this->get_fresh_graph_input_name(),
                                input_tensor_pattern),
  };

  OutputGraphExprInput output_graph_expr_input = OutputGraphExprInput{
      this->output_g.add_input(this->get_fresh_graph_input_name(),
                               std::monostate{}),
  };

  this->input_mapping.equate(pattern_input, output_graph_expr_input);

  if (name.has_value()) {
    this->pattern_input_names.equate(pattern_input, name.value());
  }

  return {
      PatternValue{pattern_input},
      OutputGraphExprValue{output_graph_expr_input},
  };
}

std::map<TensorSlotName, PatternValue> SubstitutionBuilder::add_pattern_node(
    OperatorAttributePattern const &node_pattern,
    std::map<TensorSlotName, PatternValue> const &inputs,
    std::map<TensorSlotName, TensorAttributePattern> const &output_patterns,
    std::optional<std::string> const &maybe_name) {
  KwargNodeAddedResult<TensorSlotName> node_added = this->pattern_g.add_node(
      node_pattern,
      map_values(inputs, raw_open_dataflow_value_from_pattern_value),
      output_patterns);

  if (maybe_name.has_value()) {
    std::string name = maybe_name.value();

    if (this->pattern_node_names.contains_r(name)) {
      PANIC(fmt::format("Attempted to name node {}, but a "
                        "node with that name already exists!",
                        name));
    }

    this->pattern_node_names.equate(PatternNode{node_added.node}, name);
  }

  return map_values(node_added.outputs,
                    [](KwargDataflowOutput<TensorSlotName> const &o) {
                      return pattern_value_from_raw_open_kwarg_dataflow_value(
                          OpenKwargDataflowValue<int, TensorSlotName>{o});
                    });
}

std::map<TensorSlotName, OutputGraphExprValue>
    SubstitutionBuilder::add_output_graph_node(
        OutputOperatorAttrsAssignment const &node_expr,
        std::map<TensorSlotName, OutputGraphExprValue> const &inputs,
        std::set<TensorSlotName> const &output_slots) {
  KwargNodeAddedResult<TensorSlotName> node_added = this->output_g.add_node(
      node_expr,
      map_values(inputs,
                 raw_open_kwarg_dataflow_value_from_output_graph_expr_value),
      generate_map(output_slots,
                   [](TensorSlotName) { return std::monostate{}; }));

  return map_values(
      node_added.outputs, [](KwargDataflowOutput<TensorSlotName> const &o) {
        return output_graph_expr_value_from_raw_open_kwarg_dataflow_value(
            OpenKwargDataflowValue<int, TensorSlotName>{o});
      });
}

void SubstitutionBuilder::equate_outputs(
    PatternValue const &maybe_pattern_output,
    OutputGraphExprValue const &maybe_output_graph_expr_output) {
  PatternNodeOutput pattern_output =
      maybe_pattern_output.visit<PatternNodeOutput>(overload{
          [](PatternNodeOutput const &o) { return o; },
          [&](PatternInput const &) -> PatternNodeOutput {
            PANIC(fmt::format(
                "SubstitutionBuilder::equate_outputs expected a PatternValue "
                "holding a PatternNodeOutput, but received {}",
                maybe_pattern_output));
          },
      });

  OutputGraphExprNodeOutput output_graph_expr_output =
      maybe_output_graph_expr_output.visit<OutputGraphExprNodeOutput>(overload{
          [](OutputGraphExprNodeOutput const &o) { return o; },
          [&](OutputGraphExprInput const &) -> OutputGraphExprNodeOutput {
            PANIC(fmt::format("SubstitutionBuilder::equate_outputs expected an "
                              "OutputGraphExprValue holding a "
                              "OutputGraphExprNodeOutput, but received {}",
                              maybe_output_graph_expr_output));
          },
      });

  if (this->output_mapping.contains_l(pattern_output)) {
    PANIC(
        fmt::format("SubstitutionBuilder::equate_outputs expected a "
                    "PatternValue holding a PatternValueOutput"
                    "that is not contained in the output_mapping forward graph,"
                    "but received {}",
                    pattern_output));
  }
  if (this->output_mapping.contains_r(output_graph_expr_output)) {
    PANIC(fmt::format(
        "SubstitutionBuilder::output_graph_expr_output expected a "
        "OutputGraphExprValue holding a OutputGraphExprNodeOutput"
        "that is not contained in the output_mapping backward graph,"
        "but received {}",
        output_graph_expr_output));
  }

  this->output_mapping.equate(pattern_output, output_graph_expr_output);
}

PatternNode
    SubstitutionBuilder::pattern_node_named(std::string const &name) const {
  return this->pattern_node_names.at_r(name);
}

PatternInput
    SubstitutionBuilder::pattern_input_named(std::string const &name) const {
  return this->pattern_input_names.at_r(name);
}

Substitution SubstitutionBuilder::get_substitution() const {
  Substitution result = Substitution{
      PCGPattern{this->pattern_g},
      OutputGraphExpr{this->output_g},
      this->input_mapping,
      this->output_mapping,
  };

  ASSERT(
      is_valid_substitution(result),
      "get_substitution cannot return a Substitution, as the Substitution is "
      "currently invalid. Ensure you have finished constructing the "
      "Substitution and have mapped all of the outputs.");

  return result;
}

int SubstitutionBuilder::get_fresh_graph_input_name() {
  int result = this->next_graph_input_id;
  this->next_graph_input_id++;
  return result;
}

} // namespace FlexFlow

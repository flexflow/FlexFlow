#include "substitutions/unlabelled/find_pattern_matches.h"
#include "substitutions/unlabelled/match_additional_criterion.h"
#include "substitutions/unlabelled/pattern_matching.h"
#include "substitutions/unlabelled/pattern_split.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.h"
#include "substitutions/unlabelled/unlabelled_kwarg_dataflow_graph_pattern_match.h"
#include "utils/containers/get_only.h"
#include "utils/containers/transform.h"
#include "utils/containers/unstructured_exhaustive_relational_join.h"
#include "utils/containers/values.h"
#include "utils/containers/zip.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_outgoing_kwarg_dataflow_outputs_for_node.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_inputs.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_incoming_open_kwarg_dataflow_values_for_node.h"
#include "utils/many_to_one/invert_many_to_one.h"
#include "utils/many_to_one/many_to_one_from_map.h"
#include "utils/overload.h"

namespace FlexFlow {

static std::optional<UnlabelledKwargDataflowGraphPatternMatch>
    get_candidate_singleton_match(
        UnlabelledGraphPattern const &pattern,
        OpenKwargDataflowGraphView<int, TensorSlotName> const &graph,
        Node const &graph_node,
        MatchAdditionalCriterion const &additional_criterion) {
  ASSERT(is_singleton_pattern(pattern));

  PatternNode pattern_node = get_only(get_pattern_nodes(pattern));

  if (!additional_criterion.node_criterion(pattern_node, graph_node)) {
    return std::nullopt;
  }

  UnlabelledKwargDataflowGraphPatternMatch match =
      empty_unlabelled_pattern_match();

  match.node_assignment.equate(pattern_node, graph_node);

  std::map<TensorSlotName, PatternValue> pattern_outputs =
      get_outputs_from_pattern_node(pattern, pattern_node);
  std::map<TensorSlotName, OpenKwargDataflowValue<int, TensorSlotName>>
      graph_outputs = map_values(
          get_outgoing_kwarg_dataflow_outputs_for_node(graph, graph_node),
          [](KwargDataflowOutput<TensorSlotName> const &o) {
            return OpenKwargDataflowValue<int, TensorSlotName>{o};
          });

  if (keys(pattern_outputs) != keys(graph_outputs)) {
    return std::nullopt;
  }

  for (auto const &[slot_name, pattern_val] : pattern_outputs) {
    if (!additional_criterion.value_criterion(pattern_val,
                                              graph_outputs.at(slot_name))) {
      return std::nullopt;
    }
  }

  std::map<TensorSlotName, PatternValue> pattern_node_inputs =
      get_inputs_to_pattern_node(pattern, pattern_node);
  std::set<PatternInput> pattern_graph_inputs = get_pattern_inputs(pattern);

  ASSERT(set_of(values(pattern_node_inputs)) ==
         transform(pattern_graph_inputs,
                   [](PatternInput const &i) { return PatternValue{i}; }));

  std::map<TensorSlotName, OpenKwargDataflowValue<int, TensorSlotName>>
      graph_node_inputs =
          get_incoming_open_kwarg_dataflow_values_for_node(graph, graph_node);

  if (keys(graph_node_inputs) != keys(pattern_node_inputs)) {
    return std::nullopt;
  }

  for (auto const &[slot_name, pattern_val] : pattern_node_inputs) {
    if (!additional_criterion.value_criterion(
            pattern_val, graph_node_inputs.at(slot_name))) {
      return std::nullopt;
    }
  }

  ManyToOne<TensorSlotName, PatternInput> m_pattern_node_inputs =
      many_to_one_from_map(map_values(
          pattern_node_inputs, [](PatternValue const &v) -> PatternInput {
            return v.require_pattern_input();
          }));
  ManyToOne<TensorSlotName, OpenKwargDataflowValue<int, TensorSlotName>>
      m_graph_node_inputs = many_to_one_from_map(graph_node_inputs);

  ManyToOne<PatternInput, OpenKwargDataflowValue<int, TensorSlotName>>
      input_assignment = many_to_one_from_unstructured_relation(
          unstructured_exhaustive_relational_join(
              unstructured_relation_from_one_to_many(
                  invert_many_to_one(m_pattern_node_inputs)),
              unstructured_relation_from_many_to_one(m_graph_node_inputs)));

  match.input_assignment = input_assignment.l_to_r();

  ASSERT(unlabelled_pattern_does_match(
      pattern, graph, match, additional_criterion));

  return match;
}

MatchAdditionalCriterion additional_criterion_for_subpattern(
    MatchAdditionalCriterion const &full_additional_criterion,
    bidict<PatternValue, PatternInput> const
        &full_pattern_values_to_subpattern_inputs) {
  return MatchAdditionalCriterion{
      full_additional_criterion.node_criterion,
      [&](PatternValue const &patternValue,
          OpenKwargDataflowValue<int, TensorSlotName> const &pcgValue) {
        return patternValue.visit<bool>(
            overload{[&](PatternNodeOutput const &) -> bool {
                       return full_additional_criterion.value_criterion(
                           patternValue, pcgValue);
                     },
                     [&](PatternInput const &i) -> bool {
                       PatternValue full_pattern_value =
                           full_pattern_values_to_subpattern_inputs.at_r(i);
                       return full_additional_criterion.value_criterion(
                           full_pattern_value, pcgValue);
                     }});
      }};
}

std::vector<UnlabelledKwargDataflowGraphPatternMatch>
    find_unlabelled_pattern_matches(
        UnlabelledGraphPattern const &pattern,
        OpenKwargDataflowGraphView<int, TensorSlotName> const &graph,
        MatchAdditionalCriterion const &additional_criterion) {
  std::vector<UnlabelledKwargDataflowGraphPatternMatch> matches;
  if (is_singleton_pattern(pattern)) {
    for (Node const &graph_node : get_nodes(graph)) {
      std::optional<UnlabelledKwargDataflowGraphPatternMatch> candidate =
          get_candidate_singleton_match(
              pattern, graph, graph_node, additional_criterion);
      if (candidate.has_value()) {
        matches.push_back(candidate.value());
      }
    }
  } else {
    PatternSplit split = find_even_split(pattern);
    std::set<StandardPatternEdge> split_edges =
        get_pattern_edges_across_topological_split(pattern, split);
    PatternSplitResult subpatterns = apply_split(pattern, split);
    std::vector<UnlabelledKwargDataflowGraphPatternMatch> prefix_matches =
        find_unlabelled_pattern_matches(
            subpatterns.subpattern_1,
            graph,
            additional_criterion_for_subpattern(
                additional_criterion,
                subpatterns.full_pattern_values_to_subpattern_1_inputs));
    std::vector<UnlabelledKwargDataflowGraphPatternMatch> postfix_matches =
        find_unlabelled_pattern_matches(
            subpatterns.subpattern_2,
            graph,
            additional_criterion_for_subpattern(
                additional_criterion,
                subpatterns.full_pattern_values_to_subpattern_2_inputs));

    for (UnlabelledKwargDataflowGraphPatternMatch const &prefix_match :
         prefix_matches) {
      for (UnlabelledKwargDataflowGraphPatternMatch const &postfix_match :
           postfix_matches) {
        std::optional<UnlabelledKwargDataflowGraphPatternMatch> unsplit =
            merge_unlabelled_dataflow_graph_pattern_matches(
                prefix_match,
                postfix_match,
                subpatterns.full_pattern_values_to_subpattern_1_inputs,
                subpatterns.full_pattern_values_to_subpattern_2_inputs);
        if (unsplit.has_value() &&
            unlabelled_pattern_does_match(pattern,
                                          graph,
                                          unsplit.value(),
                                          additional_criterion,
                                          split_edges)) {
          matches.push_back(unsplit.value());
        }
      }
    }
  }

  return matches;
}

} // namespace FlexFlow

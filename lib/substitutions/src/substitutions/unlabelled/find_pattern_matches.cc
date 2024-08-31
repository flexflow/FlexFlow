#include "substitutions/unlabelled/find_pattern_matches.h"
#include "substitutions/unlabelled/match_additional_criterion.h"
#include "substitutions/unlabelled/multidigraph_pattern_match.h"
#include "substitutions/unlabelled/pattern_matching.h"
#include "substitutions/unlabelled/pattern_split.h"
#include "substitutions/unlabelled/unlabelled_dataflow_graph_pattern_match.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.h"
#include "utils/containers/get_only.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_inputs.h"

namespace FlexFlow {

static std::optional<UnlabelledDataflowGraphPatternMatch>
    get_candidate_singleton_match(UnlabelledGraphPattern const &pattern,
                                  OpenDataflowGraphView const &graph,
                                  Node const &graph_node) {
  assert(is_singleton_pattern(pattern));

  PatternNode pattern_node = get_only(get_nodes(pattern));

  UnlabelledDataflowGraphPatternMatch match = empty_unlabelled_pattern_match();
  match.node_assignment.equate(pattern_node, graph_node);

  std::vector<PatternValue> pattern_outputs =
      get_outputs_from_pattern_node(pattern, pattern_node);
  std::vector<OpenDataflowValue> graph_outputs =
      transform(get_outputs(graph, graph_node),
                [](DataflowOutput const &o) { return OpenDataflowValue{o}; });

  if (pattern_outputs.size() != graph_outputs.size()) {
    return std::nullopt;
  }

  std::vector<PatternValue> pattern_node_inputs =
      get_inputs_to_pattern_node(pattern, pattern_node);
  std::unordered_set<PatternInput> pattern_graph_inputs = get_graph_inputs(pattern);

  assert(unordered_set_of(pattern_node_inputs) ==
         transform(pattern_graph_inputs,
                   [](PatternInput const &i) { return PatternValue{i}; }));

  std::vector<OpenDataflowValue> graph_node_inputs =
      get_inputs(graph, graph_node);

  if (graph_node_inputs.size() != pattern_node_inputs.size()) {
    return std::nullopt;
  }

  for (auto const &[pattern_node_input, graph_node_input] :
       zip(pattern_node_inputs, graph_node_inputs)) {
    assert(pattern_node_input.has<PatternInput>());

    match.input_assignment.insert({
        pattern_node_input.get<PatternInput>(),
        graph_node_input,
    });
  }

  assert(unlabelled_pattern_does_match(
      pattern, graph, match, match_additional_crition_always_true()));

  return match;
}

std::vector<UnlabelledDataflowGraphPatternMatch>
    find_pattern_matches(UnlabelledGraphPattern const &pattern,
                         OpenDataflowGraphView const &graph,
                         MatchAdditionalCriterion const &additional_criterion) {
  std::vector<UnlabelledDataflowGraphPatternMatch> matches;
  if (is_singleton_pattern(pattern)) {
    for (Node const &graph_node : get_nodes(graph)) {
      std::optional<UnlabelledDataflowGraphPatternMatch> candidate =
          get_candidate_singleton_match(pattern, graph, graph_node);
      if (candidate.has_value() &&
          unlabelled_pattern_does_match(
              pattern, graph, candidate.value(), additional_criterion)) {
        matches.push_back(candidate.value());
      }
    }
  } else {
    PatternSplit split = find_even_split(pattern);
    PatternSplitResult subpatterns = apply_split(pattern, split);
    std::vector<UnlabelledDataflowGraphPatternMatch> prefix_matches =
        find_pattern_matches(
            subpatterns.subpattern_1, graph, additional_criterion);
    std::vector<UnlabelledDataflowGraphPatternMatch> postfix_matches =
        find_pattern_matches(
            subpatterns.subpattern_2, graph, additional_criterion);

    for (UnlabelledDataflowGraphPatternMatch const &prefix_match :
         prefix_matches) {
      for (UnlabelledDataflowGraphPatternMatch const &postfix_match :
           postfix_matches) {
        std::optional<UnlabelledDataflowGraphPatternMatch> unsplit =
            merge_unlabelled_dataflow_graph_pattern_matches(
                prefix_match,
                postfix_match,
                subpatterns.full_pattern_values_to_subpattern_1_inputs,
                subpatterns.full_pattern_values_to_subpattern_2_inputs);
        if (unsplit.has_value() &&
            unlabelled_pattern_does_match(
                pattern, graph, unsplit.value(), additional_criterion)) {
          matches.push_back(unsplit.value());
        }
      }
    }
  }

  return matches;
}

} // namespace FlexFlow

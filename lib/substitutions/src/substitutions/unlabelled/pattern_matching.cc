#include "substitutions/unlabelled/pattern_matching.h"
#include "substitutions/unlabelled/match_split.h"
#include "substitutions/unlabelled/pattern_split.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph_inputs.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph.h"
#include <memory>

namespace FlexFlow {

std::pair<
  OpenDataflowGraphView,
  bidict<OpenDataflowValue, DataflowGraphInput>
> subgraph_matched(OpenDataflowGraphView const &g,
                                       UnlabelledDataflowGraphPatternMatch const &match) {
  std::unordered_set<Node> matched_nodes = keys(match.node_assignment.reversed());
  std::vector<OpenDataflowValue> subgraph_inputs = sorted(get_subgraph_inputs(g, matched_nodes));
  return get_subgraph(g, matched_nodes, subgraph_inputs);
}

bool unlabelled_pattern_does_match(
    UnlabelledGraphPattern const &pattern,
    OpenDataflowGraphView const &graph,
    UnlabelledDataflowGraphPatternMatch const &match,
    MatchAdditionalCriterion const &additional_criterion) {


  //   PatternNode pattern_node = get_only(get_nodes(pattern));
  //   Node matched_node = match.node_assignment.at_l(pattern_node);
  //   if (!additional_criterion.node_criterion(pattern_node, matched_node)) {
  //     return false;
  //   }
  //
  //   for (PatternValue const &pattern_value : get_values(pattern)) {
  //     OpenDataflowValue matched_value = match.value_assignment.at_l(v);
  //
  //     assert(is_input_edge(e) || is_output_edge(e));
  //     if (is_input_edge(e)) {
  //       if (is_output_edge(matched_edge)) {
  //         return false;
  //       }
  //       UpwardOpenMultiDiEdge matched_edge =
  //           narrow<UpwardOpenMultiDiEdge>(matched_edge).value();
  //       InputPatternEdge input_edge = require_input_edge(e);
  //       if (match.node_assignment.at_l(get_dst_node(input_edge)) !=
  //           get_dst_node(matched_edge)) {
  //         return false;
  //       }
  //     } else {
  //       if (is_input_edge(matched_edge)) {
  //         return false;
  //       }
  //       DownwardOpenMultiDiEdge matched_edge =
  //           narrow<DownwardOpenMultiDiEdge>(matched_edge).value();
  //       OutputPatternEdge output_edge = require_output_edge(e);
  //       if (match.node_assignment.at_l(get_src_node(output_edge)) !=
  //           get_src_node(matched_edge)) {
  //         return false;
  //       }
  //     }
  //
  //     if (!additional_criterion.value_criterion(pattern_value, matched_value)) {
  //       return false;
  //     }
  //   }
  //
  //   return true;
  // }
  //
  // PatternSplit split = find_even_split(pattern);
  // std::pair<UnlabelledGraphPattern, UnlabelledGraphPattern> subpatterns =
  //     apply_split(pattern, split);
  // auto submatches = apply_split(pattern, match, split);
  //
  // return unlabelled_pattern_does_match(subpatterns.first,
  //                                      graph,
  //                                      submatches.prefix_submatch,
  //                                      additional_criterion) &&
  //        unlabelled_pattern_does_match(subpatterns.second,
  //                                      graph,
  //                                      submatches.postfix_submatch,
  //                                      additional_criterion);
}

} // namespace FlexFlow

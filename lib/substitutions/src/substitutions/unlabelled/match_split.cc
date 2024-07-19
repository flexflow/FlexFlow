// #include "substitutions/unlabelled/match_split.h"
// #include "substitutions/unlabelled/edge_splits.h"
// #include "substitutions/unlabelled/pattern_edge.h"
// #include "substitutions/unlabelled/pattern_split.h"
// #include "substitutions/unlabelled/unlabelled_dataflow_graph_pattern_match.h"
//
// namespace FlexFlow {
//
// MatchSplit empty_match_split() {
//   return MatchSplit{empty_unlabelled_pattern_match(),
//                     empty_unlabelled_pattern_match()};
// }
//
// MatchSplit apply_split(UnlabelledGraphPattern const &pattern,
//                        UnlabelledDataflowGraphPatternMatch const &match,
//                        PatternSplit const &split) {
//   std::unordered_set<PatternNode> prefix = split.first;
//   std::unordered_set<PatternNode> postfix = split.second;
//
//   MatchSplit result = empty_match_split();
//
//   for (auto const &[pattern_node, match_node] : match.node_assignment) {
//     if (contains(split.first, pattern_node)) {
//       result.prefix_submatch.node_assignment.equate(pattern_node,
//       match_node);
//     } else {
//       assert(contains(split.second, pattern_node));
//       result.postfix_submatch.node_assignment.equate(pattern_node,
//       match_node);
//     }
//   }
//
//   UnlabelledPatternEdgeSplits edge_splits = get_edge_splits(pattern, split);
//
//   std::function<void(PatternEdge const &, OpenMultiDiEdge const &)>
//       handle_edge = [&](PatternEdge const &pattern_edge,
//                         OpenMultiDiEdge const &graph_edge) -> void {
//     std::unordered_set<PatternNode> edge_nodes = get_nodes(pattern_edge);
//
//     if (is_subseteq_of(edge_nodes, prefix)) {
//       result.prefix_submatch.edge_assignment.equate(pattern_edge,
//       graph_edge);
//     } else if (is_subseteq_of(edge_nodes, postfix)) {
//       result.postfix_submatch.edge_assignment.equate(pattern_edge,
//       graph_edge);
//     } else {
//       assert(is_standard_edge(graph_edge));
//
//       ClosedPatternEdge closed_edge = require_closed_edge(pattern_edge);
//
//       auto split = get_split_edges(edge_splits, closed_edge);
//       OutputPatternEdge output_edge = split.first;
//       InputPatternEdge input_edge = split.second;
//
//       auto split_graph_edge = split_edge(std::get<MultiDiEdge>(graph_edge));
//       OutputMultiDiEdge output_graph_edge = split_graph_edge.first;
//       InputMultiDiEdge input_graph_edge = split_graph_edge.second;
//
//       handle_edge(pattern_edge_from_input_edge(input_edge),
//                   OpenMultiDiEdge{input_graph_edge});
//       handle_edge(pattern_edge_from_output_edge(output_edge),
//                   OpenMultiDiEdge{output_graph_edge});
//     }
//   };
//
//   for (auto const &[pattern_edge, match_edge] : match.edge_assignment) {
//     handle_edge(pattern_edge, match_edge);
//   }
//
//   return result;
// }
//
// } // namespace FlexFlow

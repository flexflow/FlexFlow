#include "substitutions/unlabelled/multidigraph_pattern_match.h"
// #include "substitutions/unlabelled/edge_splits.h"
// #include "substitutions/unlabelled/pattern_edge.h"
#include "utils/containers.h"

namespace FlexFlow {

// MultiDiGraphPatternMatch empty_multidigraph_pattern_match() {
//   return MultiDiGraphPatternMatch{
//       bidict<PatternNode, Node>{},
//       bidict<PatternEdge, OpenMultiDiEdge>{},
//   };
// }

// std::optional<MultiDiGraphPatternMatch>
//     unsplit_matches(MultiDiGraphPatternMatch const &prefix,
//                     MultiDiGraphPatternMatch const &postfix,
//                     UnlabelledPatternEdgeSplits const &edge_splits) {
//
//   MultiDiGraphPatternMatch result = empty_multidigraph_pattern_match();
//
//   std::unordered_set<PatternEdge> handled;
//   for (auto const &coi : as_closed_output_input_tuples(edge_splits)) {
//     ClosedPatternEdge closed_edge = std::get<ClosedPatternEdge>(coi);
//     OutputPatternEdge output_edge = std::get<OutputPatternEdge>(coi);
//     InputPatternEdge input_edge = std::get<InputPatternEdge>(coi);
//
//     handled.insert(pattern_edge_from_output_edge(output_edge));
//     handled.insert(pattern_edge_from_input_edge(input_edge));
//
//     OpenMultiDiEdge output_graph_edge =
//         prefix.edge_assignment.at_l(pattern_edge_from_output_edge(output_edge));
//     OpenMultiDiEdge input_graph_edge =
//         postfix.edge_assignment.at_l(pattern_edge_from_input_edge(input_edge));
//     if (output_graph_edge == input_graph_edge) {
//       result.edge_assignment.equate(pattern_edge_from_closed_edge(closed_edge),
//                                     output_graph_edge);
//     } else {
//       return std::nullopt;
//     }
//   }
//
//   for (auto const &kv :
//        merge_maps(prefix.edge_assignment, postfix.edge_assignment)) {
//     if (!contains(handled, kv.first)) {
//       result.edge_assignment.equate(kv.first, kv.second);
//     }
//   }
//
//   result.node_assignment =
//       merge_maps(prefix.node_assignment, postfix.node_assignment);
//
//   return result;
// }

} // namespace FlexFlow

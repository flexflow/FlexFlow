#include "substitutions/unlabelled/edge_splits.h"

namespace FlexFlow {

// std::pair<OutputPatternEdge, InputPatternEdge>
//     get_split_edges(UnlabelledPatternEdgeSplits const &splits,
//                     ClosedPatternEdge const &e) {
//   std::pair<OutputMultiDiEdge, InputMultiDiEdge> raw_result =
//       splits.unwrapped.at_l(e.raw_edge);
//   return {
//       OutputPatternEdge{raw_result.first},
//       InputPatternEdge{raw_result.second},
//   };
// }
//
// std::vector<std::tuple<ClosedPatternEdge, OutputPatternEdge,
// InputPatternEdge>>
//     as_closed_output_input_tuples(UnlabelledPatternEdgeSplits const &s) {
//   std::vector<
//       std::tuple<ClosedPatternEdge, OutputPatternEdge, InputPatternEdge>>
//       result;
//
//   for (auto const &kv : s.unwrapped) {
//     MultiDiEdge standard_edge = kv.first;
//     OutputMultiDiEdge output_edge = kv.second.first;
//     InputMultiDiEdge input_edge = kv.second.second;
//
//     result.push_back({ClosedPatternEdge{standard_edge},
//                       OutputPatternEdge{output_edge},
//                       InputPatternEdge{input_edge}});
//   }
//
//   return result;
// }

} // namespace FlexFlow

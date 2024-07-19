#include "substitutions/unlabelled/pattern_split.h"
#include "substitutions/unlabelled/pattern_value.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph.h"

namespace FlexFlow {

PatternSplit find_even_split(UnlabelledGraphPattern const &pattern) {
  std::vector<PatternNode> topological_ordering =
      transform(get_topological_ordering(pattern.raw_graph),
                [](Node const &raw_node) { return PatternNode{raw_node}; });
  assert(topological_ordering.size() >= 2);

  int split_point = topological_ordering.size() / 2;
  auto split = vector_split(topological_ordering, split_point);
  std::unordered_set<PatternNode> prefix(split.first.begin(),
                                         split.first.end());
  std::unordered_set<PatternNode> postfix(split.second.begin(),
                                          split.second.end());
  return PatternSplit{prefix, postfix};
}

// GraphSplit get_raw_split(PatternSplit const &s) {
//   return std::pair{
//       transform(s.first, [](PatternNode const &n) { return n.raw_node; }),
//       transform(s.second, [](PatternNode const &n) { return n.raw_node; }),
//   };
// }

// UnlabelledPatternEdgeSplits
//     get_edge_splits(UnlabelledGraphPattern const &pattern,
//                     PatternSplit const &split) {
//   bidict<MultiDiEdge, std::pair<InputMultiDiEdge, OutputMultiDiEdge>>
//       raw_result = get_edge_splits(pattern.raw_graph, get_raw_split(split), );
//   return UnlabelledPatternEdgeSplits{raw_result};
// }

PatternSplitResult
    apply_split(UnlabelledGraphPattern const &p, PatternSplit const &s) {
  UnlabelledGraphPatternSubgraphResult first_subgraph_result = get_subgraph(p, s.first);
  UnlabelledGraphPatternSubgraphResult second_subgraph_result = get_subgraph(p, s.second);

  return PatternSplitResult{
    first_subgraph_result.subpattern,
    second_subgraph_result.subpattern,
    first_subgraph_result.full_pattern_values_to_subpattern_inputs,
    second_subgraph_result.full_pattern_values_to_subpattern_inputs
  };
}

} // namespace FlexFlow

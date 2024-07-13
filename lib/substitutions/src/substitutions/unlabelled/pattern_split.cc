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
  OpenDataflowSubgraphResult raw_second_subgraph_result = get_subgraph(p.raw_graph, transform(s.second, [](PatternNode const &pn) { return pn.raw_node; }));

  bidict<PatternValue, PatternInput> subpattern_1_outputs_to_subpattern_2_inputs;
  for (auto const &kv : raw_second_subgraph_result.full_graph_values_to_subgraph_inputs) {
    OpenDataflowValue open_dataflow_value = kv.first;
    DataflowGraphInput dataflow_graph_input = kv.second;
    subpattern_1_outputs_to_subpattern_2_inputs.equate(
                                                       pattern_value_from_raw_open_dataflow_value(open_dataflow_value), PatternInput{dataflow_graph_input});
  }
  
  return PatternSplitResult{
    get_subgraph(p, s.first),
    UnlabelledGraphPattern{raw_second_subgraph_result.graph},
    subpattern_1_outputs_to_subpattern_2_inputs,
  };
}

} // namespace FlexFlow

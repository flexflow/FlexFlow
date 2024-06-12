#include "substitutions/unlabelled/pattern_split.h"

namespace FlexFlow {

PatternSplit find_even_split(UnlabelledGraphPattern const &pattern) {
  std::vector<PatternNode> topological_ordering =
      transform(get_topological_ordering(pattern.raw_graph),
                [](Node const &n) { return PatternNode{n}; });
  assert(topological_ordering.size() >= 2);

  int split_point = topological_ordering.size() / 2;
  auto split = vector_split(topological_ordering, split_point);
  std::unordered_set<PatternNode> prefix(split.first.begin(),
                                         split.first.end());
  std::unordered_set<PatternNode> postfix(split.second.begin(),
                                          split.second.end());
  return PatternSplit{prefix, postfix};
}

GraphSplit get_raw_split(PatternSplit const &s) {
  return std::pair{
      transform(s.first, [](PatternNode const &n) { return n.raw_node; }),
      transform(s.second, [](PatternNode const &n) { return n.raw_node; }),
  };
}

UnlabelledPatternEdgeSplits
    get_edge_splits(UnlabelledGraphPattern const &pattern,
                    PatternSplit const &split) {
  bidict<MultiDiEdge, std::pair<OutputMultiDiEdge, InputMultiDiEdge>>
      raw_result = get_edge_splits(pattern.raw_graph, get_raw_split(split));
  return UnlabelledPatternEdgeSplits{raw_result};
}

std::pair<UnlabelledGraphPattern, UnlabelledGraphPattern>
    apply_split(UnlabelledGraphPattern const &p, PatternSplit const &s) {

  MultiDiGraphView g1 = get_subgraph(
      p.raw_graph,
      transform(s.first, [](PatternNode const &n) { return n.raw_node; }));
  MultiDiGraphView g2 = get_subgraph(
      p.raw_graph,
      transform(s.second, [](PatternNode const &n) { return n.raw_node; }));
  OpenMultiDiGraphView g1_open = as_openmultidigraph(g1);
  OpenMultiDiGraphView g2_open = as_openmultidigraph(g2);
  return std::pair{
      UnlabelledGraphPattern{g1_open},
      UnlabelledGraphPattern{g2_open},
  };
}

} // namespace FlexFlow

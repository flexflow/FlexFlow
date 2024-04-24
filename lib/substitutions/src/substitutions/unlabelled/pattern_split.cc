#include "substitutions/unlabelled/pattern_split.h"

namespace FlexFlow {

PatternSplit find_even_split(UnlabelledGraphPattern const &p) {
  std::vector<PatternNode> topological_ordering = get_topological_ordering(pattern.raw_graph);
  assert(topological_ordering.size() >= 2);

  int split_point = topological_ordering.size() / 2;
  auto split = vector_split(topological_ordering, split_point);
  std::unordered_set<PatternNode> prefix(split.first.begin(), split.first.end());
  std::unordered_set<PatternNode> postfix(split.second.begin(), split.second.end());
  return {prefix, postfix};
}

GraphSplit get_raw_split(PatternSplit const &s) {
  return std::pair{
    transform(s.first, [](PatternNode const &n) { return n.raw_node; }),
    transform(s.second, [](PatternNode const &n) { return n.raw_node; }),
  };
}

UnlabelledPatternEdgeSplits get_edge_splits(UnlabelledGraphPattern const &pattern, PatternSplit const &split) {
  bidict<MultiDiEdge, std::pair<InputMultiDiEdge, OutputMultiDiEdge>> raw_result = get_edge_splits(
    pattern.raw_graph, 
    get_raw_split(split),
  );
  return UnlabelledPatternEdgeSplits{raw_result};
}

std::pair<UnlabelledGraphPattern, UnlabelledGraphPattern>
  apply_split(UnlabelledGraphPattern const &p, PatternSplit const &s) {
  return {
    get_subgraph(p, s.left);
    get_subgraph(p, s.right);
  };
}

} // namespace FlexFlow

#include "substitutions/unlabelled/pattern_split.h"
#include "substitutions/unlabelled/pattern_value.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph.h"
#include "utils/containers/vector_split.h"

namespace FlexFlow {

PatternSplit find_even_split(UnlabelledGraphPattern const &pattern) {
  std::vector<PatternNode> topological_ordering = get_topological_ordering(pattern);
  assert(topological_ordering.size() >= 2);

  int split_point = topological_ordering.size() / 2;
  auto split = vector_split(topological_ordering, split_point);
  std::unordered_set<PatternNode> prefix = unordered_set_of(split.first);
  std::unordered_set<PatternNode> postfix = unordered_set_of(split.second);
  return PatternSplit{prefix, postfix};
}

PatternSplitResult apply_split(UnlabelledGraphPattern const &p,
                               PatternSplit const &s) {
  UnlabelledGraphPatternSubgraphResult first_subgraph_result =
      get_subgraph(p, s.first);
  UnlabelledGraphPatternSubgraphResult second_subgraph_result =
      get_subgraph(p, s.second);

  return PatternSplitResult{
      first_subgraph_result.subpattern,
      second_subgraph_result.subpattern,
      first_subgraph_result.full_pattern_values_to_subpattern_inputs,
      second_subgraph_result.full_pattern_values_to_subpattern_inputs};
}

} // namespace FlexFlow

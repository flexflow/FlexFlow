#include "substitutions/unlabelled/unlabelled_dataflow_graph_pattern_match.h"

namespace FlexFlow {

UnlabelledDataflowGraphPatternMatch empty_unlabelled_pattern_match() {
  return UnlabelledDataflowGraphPatternMatch{
    bidict<PatternNode, Node>{},
    bidict<PatternInput, DataflowGraphInput>{},
  };
}

template <typename L, typename R>
std::optional<bidict<L, R>> try_merge_nondisjoint_bidicts(bidict<L, R> const &d1,
                                                          bidict<L, R> const &d2) {
  for (L const &l : intersection(keys(d1), keys(d2))) {
    return  
  }
}


std::optional<UnlabelledDataflowGraphPatternMatch>
  merge_unlabelled_dataflow_graph_pattern_matches(UnlabelledDataflowGraphPatternMatch const &subpattern_1,
                                                UnlabelledDataflowGraphPatternMatch const &subpattern_2,
                                                bidict<PatternValue, PatternInput> const &outputs_of_1_to_inputs_of_2) {
  if (!are_disjoint(matched_nodes(subpattern_1), matched_nodes(subpattern_2))) {
    return std::nullopt;
  }

  bidict<PatternNode, Node> merged_node_assignment = merge_maps(subpattern_1.node_assignment, subpattern_2.node_assignment);

  // if (!are_disjoint(matched_values(subpattern_1), matched_values(subpattern_2))) {
  //
  // }
}

} // namespace FlexFlow

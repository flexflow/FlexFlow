#include "substitutions/unlabelled/unlabelled_dataflow_graph_pattern_match.h"
#include "utils/containers.h"

namespace FlexFlow {

UnlabelledDataflowGraphPatternMatch empty_unlabelled_pattern_match() {
  return UnlabelledDataflowGraphPatternMatch{
    bidict<PatternNode, Node>{},
    bidict<PatternInput, OpenDataflowValue>{},
  };
}

template <typename L, typename R>
std::optional<bidict<L, R>> try_merge_nondisjoint_bidicts(bidict<L, R> const &d1,
                                                          bidict<L, R> const &d2) {
  bidict<L, R> result;
  for (L const &l : set_union(keys(d1), keys(d2))) {
    if (d1.contains_l(l) && d2.contains_l(l)) {
      if (d1.at_l(l) == d2.at_l(l)) {
        result.equate(l, d1.at_l(l));
      } else {
        return std::nullopt;
      }
    } else if (d1.contains_l(l)) {
      result.equate(l, d1.at_l(l));
    } else {
      assert (d2.contains_l(l));

      result.equate(l, d2.at_l(l));
    }
  }
        
  return result;
}


std::optional<UnlabelledDataflowGraphPatternMatch>
  merge_unlabelled_dataflow_graph_pattern_matches(UnlabelledDataflowGraphPatternMatch const &subpattern_1,
                                                UnlabelledDataflowGraphPatternMatch const &subpattern_2,
                                                bidict<PatternValue, PatternInput> const &outputs_of_1_to_inputs_of_2) {
  bidict<PatternNode, Node> merged_node_assignment = ({
    std::optional<bidict<PatternNode, Node>> result = try_merge_nondisjoint_bidicts(
      subpattern_1.node_assignment, subpattern_2.node_assignment);
    if (!result.has_value()) {
      return std::nullopt;
    }
    result.value();
  });

  assert (all_of(keys(subpattern_2.input_assignment), [&](PatternInput const &i) { return outputs_of_1_to_inputs_of_2.contains_r(i); }));

  return UnlabelledDataflowGraphPatternMatch{
    merged_node_assignment,
    subpattern_1.input_assignment,
  };
}

} // namespace FlexFlow

#include "substitutions/unlabelled/unlabelled_dataflow_graph_pattern_match.h"
#include "utils/bidict/try_merge_nondisjoint_bidicts.h"
#include "utils/containers.h"
#include "utils/containers/filtermap_keys.h"
#include "utils/containers/try_merge_nondisjoint_unordered_maps.h"

namespace FlexFlow {

UnlabelledDataflowGraphPatternMatch empty_unlabelled_pattern_match() {
  return UnlabelledDataflowGraphPatternMatch{
      bidict<PatternNode, Node>{},
      bidict<PatternInput, OpenDataflowValue>{},
  };
}

std::optional<UnlabelledDataflowGraphPatternMatch>
    merge_unlabelled_dataflow_graph_pattern_matches(
        UnlabelledDataflowGraphPatternMatch const &subpattern_1,
        UnlabelledDataflowGraphPatternMatch const &subpattern_2,
        bidict<PatternValue, PatternInput> const
            &merged_graph_values_to_inputs_of_1,
        bidict<PatternValue, PatternInput> const
            &merged_graph_values_to_inputs_of_2) {
  bidict<PatternNode, Node> merged_node_assignment = ({
    std::optional<bidict<PatternNode, Node>> result =
        try_merge_nondisjoint_bidicts(subpattern_1.node_assignment,
                                      subpattern_2.node_assignment);
    if (!result.has_value()) {
      return std::nullopt;
    }
    result.value();
  });

  std::unordered_map<PatternInput, OpenDataflowValue> merged_input_assignment =
      ({
        std::unordered_map<PatternValue, OpenDataflowValue>
            lifted_input_assignment_1 = map_keys(
                subpattern_1.input_assignment, [&](PatternInput const &pi1) {
                  return merged_graph_values_to_inputs_of_1.at_r(pi1);
                });
        std::unordered_map<PatternValue, OpenDataflowValue>
            lifted_input_assignment_2 = map_keys(
                subpattern_2.input_assignment, [&](PatternInput const &pi2) {
                  return merged_graph_values_to_inputs_of_2.at_r(pi2);
                });
        std::optional<std::unordered_map<PatternValue, OpenDataflowValue>>
            merged = try_merge_nondisjoint_unordered_maps(
                lifted_input_assignment_1, lifted_input_assignment_2);
        if (!merged.has_value()) {
          return std::nullopt;
        }
        filtermap_keys(
            merged.value(),
            [](PatternValue const &v) -> std::optional<PatternInput> {
              if (v.has<PatternInput>()) {
                return v.get<PatternInput>();
              } else {
                return std::nullopt;
              }
            });
      });

  return UnlabelledDataflowGraphPatternMatch{
      merged_node_assignment,
      merged_input_assignment,
  };
}

} // namespace FlexFlow

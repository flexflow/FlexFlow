#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_UNLABELLED_DATAFLOW_GRAPH_PATTERN_MATCH_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_UNLABELLED_DATAFLOW_GRAPH_PATTERN_MATCH_H

#include "substitutions/unlabelled/pattern_value.dtg.h"
#include "substitutions/unlabelled/unlabelled_dataflow_graph_pattern_match.dtg.h"
#include <unordered_set>
#include <optional>

namespace FlexFlow {

UnlabelledDataflowGraphPatternMatch empty_unlabelled_pattern_match();
std::unordered_set<Node> matched_nodes(UnlabelledDataflowGraphPatternMatch const &);
std::optional<UnlabelledDataflowGraphPatternMatch> merge_unlabelled_dataflow_graph_pattern_matches(UnlabelledDataflowGraphPatternMatch const &subpattern_1,
                                                UnlabelledDataflowGraphPatternMatch const &subpattern_2,
                                                bidict<PatternValue, PatternInput> const &merged_graph_values_to_inputs_of_1,
                                                bidict<PatternValue, PatternInput> const &merged_graph_values_to_inputs_of_2);

} // namespace FlexFlow

#endif

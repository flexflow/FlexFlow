#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_PATTERN_MATCHING_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_PATTERN_MATCHING_H

#include "substitutions/unlabelled/match_additional_criterion.dtg.h"
#include "substitutions/unlabelled/match_split.dtg.h"
#include "substitutions/unlabelled/unlabelled_dataflow_graph_pattern_match.dtg.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.dtg.h"
#include "utils/graph.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"

namespace FlexFlow {

// OpenDataflowGraphView apply_match(UnlabelledGraphPattern const &pattern,
//                                   UnlabelledDataflowGraphPatternMatch const &match);

OpenDataflowGraphView subgraph_matched(UnlabelledGraphPattern const &pattern,
                                       UnlabelledDataflowGraphPatternMatch const &match);

bool unlabelled_pattern_does_match(
    UnlabelledGraphPattern const &pattern,
    OpenDataflowGraphView const &graph,
    UnlabelledDataflowGraphPatternMatch const &match,
    MatchAdditionalCriterion const &additional_criterion);

std::vector<UnlabelledDataflowGraphPatternMatch>
    find_pattern_matches(UnlabelledGraphPattern const &pattern,
                         OpenDataflowGraphView const &graph,
                         MatchAdditionalCriterion const &additional_criterion);

} // namespace FlexFlow

#endif

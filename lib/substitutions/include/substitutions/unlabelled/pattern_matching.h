#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_PATTERN_MATCHING_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_PATTERN_MATCHING_H

#include "substitutions/unlabelled/match_additional_criterion.dtg.h"
#include "substitutions/unlabelled/unlabelled_dataflow_graph_pattern_match.dtg.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.dtg.h"
#include "utils/graph.h"
#include "utils/graph/open_dataflow_graph/algorithms/open_dataflow_subgraph_result.dtg.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"

namespace FlexFlow {

// OpenDataflowGraphView apply_match(UnlabelledGraphPattern const &pattern,
//                                   UnlabelledDataflowGraphPatternMatch const
//                                   &match);

OpenDataflowSubgraphResult
    subgraph_matched(OpenDataflowGraphView const &graph,
                     UnlabelledDataflowGraphPatternMatch const &match);
bool pattern_matches_subgraph_under(
    UnlabelledGraphPattern const &pattern,
    OpenDataflowGraphView const &subgraph,
    bidict<OpenDataflowValue, DataflowGraphInput> const
        &full_graph_values_to_subgraph_inputs,
    UnlabelledDataflowGraphPatternMatch const &match,
    MatchAdditionalCriterion const &additional_criterion);

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

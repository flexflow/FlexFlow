#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_PATTERN_MATCHING_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_PATTERN_MATCHING_H

#include "substitutions/unlabelled/match_additional_criterion.dtg.h"
#include "substitutions/unlabelled/match_split.dtg.h"
#include "substitutions/unlabelled/multidigraph_pattern_match.dtg.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.dtg.h"
#include "utils/graph.h"

namespace FlexFlow {

bool unlabelled_pattern_does_match(
    UnlabelledGraphPattern const &pattern,
    OpenMultiDiGraphView const &graph,
    MultiDiGraphPatternMatch const &match,
    MatchAdditionalCriterion const &additional_criterion);

std::vector<MultiDiGraphPatternMatch>
    find_pattern_matches(UnlabelledGraphPattern const &pattern,
                         OpenMultiDiGraphView const &graph,
                         MatchAdditionalCriterion const &additional_criterion);

} // namespace FlexFlow

#endif

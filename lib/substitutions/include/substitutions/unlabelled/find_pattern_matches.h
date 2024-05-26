#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_FIND_PATTERN_MATCHES_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_FIND_PATTERN_MATCHES_H

#include "substitutions/unlabelled/match_additional_criterion.dtg.h"
#include "substitutions/unlabelled/multidigraph_pattern_match.dtg.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.dtg.h"
#include "utils/graph.h"

namespace FlexFlow {

std::vector<MultiDiGraphPatternMatch>
    find_pattern_matches(UnlabelledGraphPattern const &pattern,
                         OpenMultiDiGraphView const &graph,
                         MatchAdditionalCriterion const &additional_criterion);

} // namespace FlexFlow

#endif

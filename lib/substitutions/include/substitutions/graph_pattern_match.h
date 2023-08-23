#ifndef _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_MATCH_H
#define _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_MATCH_H

#include "utils/graph.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct DiGraphPatternMatch {
  bidict<Node, Node> nodeAssignment;
  bidict<OpenMultiDiEdge, MultiDiEdge> edgeAssignment;
};

struct MatchSplit {
  DiGraphPatternMatch prefix_submatch;
  DiGraphPatternMatch postfix_submatch;
};

template <typename F>
bool pattern_matches(OpenMultiDiGraphView const &,
                     MultiDiGraphView const &,
                     DiGraphPatternMatch const &,
                     F const &additional_criterion);

} // namespace FlexFlow

#endif

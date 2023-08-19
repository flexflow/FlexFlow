#ifndef _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_H
#define _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_H

#include "utils/graph.h"

namespace FlexFlow {

struct DiGraphPatternMatch {
  bidict<Node, Node> nodeAssignment;
  req<bidict<OpenMultiDiEdge, MultiDiEdge>> edgeAssignment;
};

FF_VISITABLE_STRUCT(DiGraphPatternMatch, nodeAssignment, edgeAssignment);

struct MatchSplit {
  DiGraphPatternMatch prefix_submatch;
  req<DiGraphPatternMatch> postfix_submatch;
};

FF_VISITABLE_STRUCT(MatchSplit, prefix_submatch, postfix_submatch);

GraphSplit split_pattern(OpenMultiDiGraphView const &pattern);

bool pattern_matches(OpenMultiDiGraphView const &,
                     MultiDiGraphView const &,
                     DiGraphPatternMatch const &,
                     F const &additional_criterion);

bool is_singleton_pattern(OpenMultiDiGraphView const &);

} // namespace FlexFlow

#endif

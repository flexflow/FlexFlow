#ifndef _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_H
#define _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_H

#include "utils/graph.h"

namespace FlexFlow {
namespace substitutions {

struct DiGraphPatternMatch {
  bidict<Node, Node> nodeAssignment;
  bidict<OpenMultiDiEdge, MultiDiEdge> edgeAssignment;
};

struct MatchSplit {
  DiGraphPatternMatch prefix_submatch;
  DiGraphPatternMatch postfix_submatch;
};

GraphSplit split_pattern(IOpenMultiDiGraph const &pattern);

bool pattern_matches(IOpenMultiDiGraphView const &,
                     IMultiDiGraph const &,
                     DiGraphPatternMatch const &);
bool is_singleton_pattern(IOpenMultiDiGraphView const &);

} // namespace substitutions
} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::substitutions::DiGraphPatternMatch> {
  size_t
      operator()(::FlexFlow::substitutions::DiGraphPatternMatch const &) const;
};

} // namespace std

#endif

#ifndef _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_MATCH_H
#define _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_MATCH_H

#include "utils/graph.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct MultiDiGraphPatternMatch {
  using PatternNode = Node;
  using PCGNode = Node;
  using PatternEdge = OpenMultiDiEdge;
  using PCGEdge = MultiDiEdge;

  bidict<PatternNode, PCGNode> nodeAssignment;
  bidict<PatternEdge, PCGEdge> edgeAssignment;
};

struct MatchSplit {
  MultiDiGraphPatternMatch prefix_submatch;
  MultiDiGraphPatternMatch postfix_submatch;
};

template <typename F>
bool pattern_matches(OpenMultiDiGraphView const &,
                     MultiDiGraphView const &,
                     MultiDiGraphPatternMatch const &,
                     F const &additional_criterion);

template <typename F>
std::unordered_set<MultiDiGraphPatternMatch>
    find_pattern_matches(OpenMultiDiGraphView const &pattern,
                         MultiDiGraphView const &graph,
                         F const &additional_criterion);

} // namespace FlexFlow

#endif

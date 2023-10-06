#ifndef _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_MATCH_H
#define _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_MATCH_H

#include "utils/graph.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct MultiDiGraphPatternMatch {
  using PatternNode = Node;
  using PCGNode = Node;
  using PatternEdge = OpenMultiDiEdge;
  using PCGEdge = OpenMultiDiEdge;

  bidict<PatternNode, PCGNode> node_assignment;
  bidict<PatternEdge, PCGEdge> edge_assignment;
};

struct MatchSplit {
  MultiDiGraphPatternMatch prefix_submatch;
  MultiDiGraphPatternMatch postfix_submatch;
};

template <typename F>
bool pattern_matches(OpenMultiDiGraphView const &pattern,
                     OpenMultiDiGraphView const &graph,
                     MultiDiGraphPatternMatch const &match,
                     F const &additional_criterion);

template <typename F>
std::unordered_set<MultiDiGraphPatternMatch>
    find_pattern_matches(OpenMultiDiGraphView const &pattern,
                         OpenMultiDiGraphView const &graph,
                         F const &additional_criterion);

} // namespace FlexFlow

#endif

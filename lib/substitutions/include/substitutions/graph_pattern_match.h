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

struct MatchAdditionalCriterion {
  std::function<bool(Node const &, Node const &)> node_criterion;
  std::function<bool(OpenMultiDiEdge const &, OpenMultiDiEdge const &)>
      edge_criterion;
};

bool pattern_matches(OpenMultiDiGraphView const &pattern,
                     OpenMultiDiGraphView const &graph,
                     MultiDiGraphPatternMatch const &match,
                     MatchAdditionalCriterion const &additional_criterion);

std::vector<MultiDiGraphPatternMatch>
    find_pattern_matches(OpenMultiDiGraphView const &pattern,
                         OpenMultiDiGraphView const &graph,
                         MatchAdditionalCriterion const &additional_criterion);

} // namespace FlexFlow

#endif

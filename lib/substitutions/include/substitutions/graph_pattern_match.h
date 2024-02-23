#ifndef _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_MATCH_H
#define _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_MATCH_H

#include "utils/graph.h"
#include "utils/visitable.h"

namespace FlexFlow {

/**
 * @struct MultiDiGraphPatternMatch
 * @brief MultiDiGraphPatternMatch is a struct that describes a mapping from how an open graph is matched with
 *  a PCG graph.
 * To apply a substitution to a PCG, we should first match the pattern graph to a subgraph of the PCG. MultiDiGraphPatternMatch describes the match, 
 * which consists of a node_assignment that describes how the GraphPattern node mapped to PCG node and an edge_assignment that describes
 * how the GraphPattern edge mapped to PCG edge.
 */
struct MultiDiGraphPatternMatch {
  using PatternNode = Node;
  using PCGNode = Node;

  /**
   * @see OpenMultiDiEdge
   */
  using PatternEdge = OpenMultiDiEdge;
  using PCGEdge = OpenMultiDiEdge;

  /**
   * @brief node_assignment is a bidirectional map from PatternNode to PCGNode
   */
  bidict<PatternNode, PCGNode> node_assignment;

  /**
   * @brief edge_assignment is a bidirectional map from PatternEdge to PCGEdge
   */
  bidict<PatternEdge, PCGEdge> edge_assignment;
};

/**
 * @struct MatchSplit
 * @brief MatchSplit is a struct that describes a split of a MultiDiGraphPatternMatch into two sub MultiDiGraphPatternMatch
 * 
 */
struct MatchSplit {
  MultiDiGraphPatternMatch prefix_submatch;
  MultiDiGraphPatternMatch postfix_submatch;
};

/**
 * @struct MatchAdditionalCriterion
 * @brief The additional conditions need to be satisfied other than geometric properties of the graph.
 */
struct MatchAdditionalCriterion {
  std::function<bool(Node const &, Node const &)> node_criterion;
  std::function<bool(OpenMultiDiEdge const &, OpenMultiDiEdge const &)>
      edge_criterion;
};

/**
 * @brief pattern_matches checks if the pattern graph matches the graph with additional conditions defined by additional_criterion.
 * @param pattern The pattern graph
 * @param graph The graph to be matched
 * @param match The mapping between the pattern graph and the graph
 * @param additional_criterion The additional conditions need to be satisfied other than geometric properties of the graph.
 * @return true if the pattern graph matches the graph, false otherwise.
 * @details function is used to check whether the generated match from pattern to graph is valid or not. It is used in find_pattern_matches to check against all the enumerated matches
 * and filter out the invalid ones.
 */
bool pattern_matches(OpenMultiDiGraphView const &pattern,
                     OpenMultiDiGraphView const &graph,
                     MultiDiGraphPatternMatch const &match,
                     MatchAdditionalCriterion const &additional_criterion);

/**
 * @brief generate all valid matches from pattern to a subgraph of graph
 * @param pattern 
 * @param graph 
 * @param additional_criterion 
 * @return std::vector<MultiDiGraphPatternMatch> 
 * 
 * @details Given a pattern and a graph, find all the valid matches between the pattern and the graph with additional conditions defined by additional_criterion.
 */ 
std::vector<MultiDiGraphPatternMatch>
    find_pattern_matches(OpenMultiDiGraphView const &pattern,
                         OpenMultiDiGraphView const &graph,
                         MatchAdditionalCriterion const &additional_criterion);

} // namespace FlexFlow

#endif

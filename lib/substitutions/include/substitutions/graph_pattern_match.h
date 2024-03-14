#ifndef _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_MATCH_H
#define _FLEXFLOW_SUBSTITUTIONS_GRAPH_PATTERN_MATCH_H

#include "utils/graph.h"
#include "utils/visitable.h"

namespace FlexFlow {

/**
 * @struct MultiDiGraphPatternMatch
 * @brief MultiDiGraphPatternMatch describes a specific location in an OpenMultiDiGraph where a given pattern matches.
 * 
 * Given a graph and a pattern there can be zero, one, or multiple locations where it can match.
 * 
 * To provide some intuition, consider matching over strings instead of graphs: given a regex pattern "a.b" and a string "acbfadbga", there are two valid match locations: 
 * we can either match the "acb" at the beginning of the string, or the "adb" in the middle of the string.
 * MultiDiGraphPatternMatch represents the difference between the two possible locations using a bidict which maps between 
 * objects in the pattern and the corresponding objects in the matched data structure. For example, in the string example above,
 * the two matchings would be as follows:
 * "acbfadbga"   "acbfadbga"
 *  ^^^               ^^^
 *  |||               |||
 *  vvv               vvv
 * "a.b"             "a.b"
 * Of course in the context of graphs there are two types of objects to be matched: nodes and edges. 
 * As such our match consists of not one but two bidict mappings: one for nodes (node_assignment) and one for edges (edge_assignment).
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
   * @brief node_assignment describes the mapping between PatternNode and PCGNode as a part of the substitution.
   */
  bidict<PatternNode, PCGNode> node_assignment;

  /**
   * @brief edge_assignment describes the mapping between PatternEdge and PCGEdge as a part of the substitution.
   */
  bidict<PatternEdge, PCGEdge> edge_assignment;
};

/**
 * @struct MatchSplit
 * @brief MatchSplit is a struct that describes a split of a MultiDiGraphPatternMatch into 
 * two sub MultiDiGraphPatternMatches by dividing the nodes into half. When applying pattern 
 * matches, the pattern will be split into two parts and recursively matched against the graph.
 */
struct MatchSplit {
  MultiDiGraphPatternMatch prefix_submatch;
  MultiDiGraphPatternMatch postfix_submatch;
};

/**
 * @struct MatchAdditionalCriterion
 * @brief The additional conditions need to be satisfied other than geometric properties of the graph.
 * Specifically as mentioned in attribute_expr.h, other than matching graph topology, we also need to make sure 
 * the attributes(eg. shape of dense layer) should be matched as well. The additional constraints
 * AttributeConstraint will be imposed inside node_criterion and edge_criterion for each potential match.
 */
struct MatchAdditionalCriterion {
  std::function<bool(Node const &, Node const &)> node_criterion;
  std::function<bool(OpenMultiDiEdge const &, OpenMultiDiEdge const &)>
      edge_criterion;
};

/**
 * @brief pattern_matches checks if the pattern graph matches the graph with additional conditions defined 
 * by additional_criterion. It is used as the last checking step to see if the pattern matches the graph 
 * attributewise inside find_pattern_matches.
 */
bool pattern_matches(OpenMultiDiGraphView const &pattern,
                     OpenMultiDiGraphView const &graph,
                     MultiDiGraphPatternMatch const &match,
                     MatchAdditionalCriterion const &additional_criterion);

/**
 * @brief find_pattern_matches generate all valid matches from pattern to a subgraph of graph.
 */ 
std::vector<MultiDiGraphPatternMatch>
    find_pattern_matches(OpenMultiDiGraphView const &pattern,
                         OpenMultiDiGraphView const &graph,
                         MatchAdditionalCriterion const &additional_criterion);

} // namespace FlexFlow

#endif

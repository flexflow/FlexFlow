#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_DOMINATORS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_DOMINATORS_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

/**
 * @brief See https://en.wikipedia.org/wiki/Dominator_(graph_theory)
 *
 * @note By definition, the root node dominates every node and every node
 * dominates itself.
 *
 */
std::unordered_set<Node> get_dominators(DiGraphView const &, Node const &);

/**
 * @brief Returns the intersection of the dominators of the given set of nodes.
 * @note This is conceptually equivalent to merging the given set of nodes and
 * then finding the set of dominators of the new merged node (where merged means
 * that all edges belonging to the set of nodes now pass through a single
 * unified node).
 */
std::unordered_set<Node> get_dominators(DiGraphView const &,
                                        std::unordered_set<Node> const &);

} // namespace FlexFlow

#endif

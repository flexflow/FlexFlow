#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_DOMINATORS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_DOMINATORS_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

/**
 * @brief A node `d` is said to dominate a node `n` if every path from the root
 * node to `n` must go through `d`.
 *
 * @note By definition, the root node dominates every node and every node
 * dominates itself.
 *
 */
std::unordered_set<Node> get_dominators(DiGraphView const &, Node const &);

/**
 * @brief Returns the intersection of the dominators of the input nodes.
 *
 */
std::unordered_set<Node> get_dominators(DiGraphView const &,
                                        std::unordered_set<Node> const &);

} // namespace FlexFlow

#endif

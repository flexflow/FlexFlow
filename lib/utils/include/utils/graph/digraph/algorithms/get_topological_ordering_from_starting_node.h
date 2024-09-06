#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_TOPOLOGICAL_ORDERING_FROM_STARTING_NODE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_TOPOLOGICAL_ORDERING_FROM_STARTING_NODE_H

#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/node/node.dtg.h"

namespace FlexFlow {

/**
 * @brief Returns a topologically ordered vector of nodes, with the topological
 * traversal starting from the starting node.
 *
 * @note Nodes present within the graph that are not reachable by a traversal
 * starting from the starting_node will not be included in the returned vector.
 *       g must be an acyclic graph
 */
std::vector<Node>
    get_topological_ordering_from_starting_node(DiGraphView const &g,
                                                Node const &starting_node);

} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_TOPOLOGICAL_ORDERING_FROM_STARTING_NODE_H

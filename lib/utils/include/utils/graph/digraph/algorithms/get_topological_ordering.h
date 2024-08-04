#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_TOPOLOGICAL_ORDERING_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_TOPOLOGICAL_ORDERING_H

#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/node/node.dtg.h"
namespace FlexFlow {

std::vector<Node> get_topological_ordering(DiGraphView const &);

std::vector<Node>
    get_topological_ordering_from_starting_node(DiGraphView const &g,
                                                Node const &starting_node);

} // namespace FlexFlow

#endif

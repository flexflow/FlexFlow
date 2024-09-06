#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_ALGORITHMS_GET_NEIGHBORING_NODES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_ALGORITHMS_GET_NEIGHBORING_NODES_H

#include "utils/graph/undirected/undirected_graph_view.h"

namespace FlexFlow {

std::unordered_set<Node> get_neighboring_nodes(UndirectedGraphView const &,
                                               Node const &);

} // namespace FlexFlow

#endif

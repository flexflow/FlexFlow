#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_ALGORITHMS_GET_CONNECTED_COMPONENTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_ALGORITHMS_GET_CONNECTED_COMPONENTS_H

#include "utils/graph/undirected/undirected_graph_view.h"

namespace FlexFlow {

std::unordered_set<std::unordered_set<Node>>
    get_connected_components(UndirectedGraphView const &);

} // namespace FlexFlow

#endif

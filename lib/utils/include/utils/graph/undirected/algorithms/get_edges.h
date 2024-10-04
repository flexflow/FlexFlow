#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_ALGORITHMS_GET_EDGES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_ALGORITHMS_GET_EDGES_H

#include "utils/graph/undirected/undirected_graph_view.h"

namespace FlexFlow {

std::unordered_set<UndirectedEdge> get_edges(UndirectedGraphView const &);

} // namespace FlexFlow

#endif

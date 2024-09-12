#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_UNDIRECTED_EDGE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_UNDIRECTED_EDGE_H

#include "utils/graph/node/node.dtg.h"
#include "utils/graph/undirected/undirected_edge.dtg.h"

namespace FlexFlow {

bool is_connected_to(UndirectedEdge const &e, Node const &n);

} // namespace FlexFlow

#endif

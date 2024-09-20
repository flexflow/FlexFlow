#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_ALGORITHMS_ADD_EDGES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_ALGORITHMS_ADD_EDGES_H

#include "utils/graph/multidigraph/multidigraph.h"

namespace FlexFlow {

std::vector<MultiDiEdge> add_edges(MultiDiGraph &,
                                   std::vector<std::pair<Node, Node>> const &);

} // namespace FlexFlow

#endif

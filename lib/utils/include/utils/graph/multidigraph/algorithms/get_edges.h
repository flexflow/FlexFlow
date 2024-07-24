#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_ALGORITHMS_GET_EDGES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_ALGORITHMS_GET_EDGES_H

#include "utils/graph/multidigraph/multidigraph_view.h"

namespace FlexFlow {

std::unordered_set<MultiDiEdge> get_edges(MultiDiGraphView const &);

} // namespace FlexFlow

#endif

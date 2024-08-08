#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_ALGORITHMS_GET_OUTGOING_EDGES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_ALGORITHMS_GET_OUTGOING_EDGES_H

#include "utils/graph/multidigraph/multidigraph_view.h"

namespace FlexFlow {

std::unordered_set<MultiDiEdge> get_outgoing_edges(MultiDiGraphView const &,
                                                   Node const &);

} // namespace FlexFlow

#endif

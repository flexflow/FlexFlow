#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_ALGORITHMS_GET_EDGE_COUNTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_ALGORITHMS_GET_EDGE_COUNTS_H

#include "utils/graph/multidigraph/multidigraph_view.h"

namespace FlexFlow {

std::unordered_map<DirectedEdge, int> get_edge_counts(MultiDiGraphView const &);

} // namespace FlexFlow

#endif

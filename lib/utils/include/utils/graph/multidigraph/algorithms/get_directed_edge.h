#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_ALGORITHMS_GET_DIRECTED_EDGE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_ALGORITHMS_GET_DIRECTED_EDGE_H

#include "utils/graph/multidigraph/multidigraph_view.h"

namespace FlexFlow {

DirectedEdge get_directed_edge(MultiDiGraphView const &, MultiDiEdge const &);

} // namespace FlexFlow

#endif

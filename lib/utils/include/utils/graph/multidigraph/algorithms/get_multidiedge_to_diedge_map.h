#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_ALGORITHMS_GET_MULTIDIEDGE_TO_DIEDGE_MAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_ALGORITHMS_GET_MULTIDIEDGE_TO_DIEDGE_MAP_H

#include "utils/graph/multidigraph/multidigraph_view.h"

namespace FlexFlow {

std::unordered_map<MultiDiEdge, DirectedEdge>
    get_multidiedge_to_diedge_map(MultiDiGraphView const &);

} // namespace FlexFlow

#endif

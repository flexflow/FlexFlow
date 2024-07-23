#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_WEAKLY_CONNECTED_COMPONENTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_WEAKLY_CONNECTED_COMPONENTS_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

std::unordered_set<std::unordered_set<Node>>
    get_weakly_connected_components(DiGraphView const &);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_DESCENDANTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_DESCENDANTS

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

std::unordered_set<Node> get_descendants(DiGraphView const &g,
                                         Node const &starting_node);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_DIGRAPH_AS_DOT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_DIGRAPH_AS_DOT_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

std::string digraph_as_dot(
    DiGraphView const &,
    std::function<std::string(Node const &)> const &get_node_label);

} // namespace FlexFlow

#endif

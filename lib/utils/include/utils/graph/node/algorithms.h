#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_NODE_ALGORITHMS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_NODE_ALGORITHMS_H

#include "utils/graph/node/graph_view.h"

namespace FlexFlow {

std::unordered_set<Node> get_nodes(GraphView const &);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_NODE_WITH_GREATEST_TOPO_RANK_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_NODE_WITH_GREATEST_TOPO_RANK_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

Node get_node_with_greatest_topo_rank(std::unordered_set<Node> const &,
                                      DiGraphView const &);

} // namespace FlexFlow

#endif

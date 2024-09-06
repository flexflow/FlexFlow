#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_SUBGRAPH_OUTGOING_EDGES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_SUBGRAPH_OUTGOING_EDGES_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

std::unordered_set<DirectedEdge> get_subgraph_outgoing_edges(DiGraphView const &,
                                                             std::unordered_set<Node> const &);

} // namespace FlexFlow

#endif

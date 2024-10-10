#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_EDGES_FROM_SUBGRAPH_TO_SUBGRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_EDGES_FROM_SUBGRAPH_TO_SUBGRAPH_H

#include "utils/graph/digraph/digraph_view.h"
namespace FlexFlow {

std::unordered_set<DirectedEdge>
    get_edges_from_subgraph_to_subgraph(DiGraphView const &,
                                        std::unordered_set<Node> const &,
                                        std::unordered_set<Node> const &);

} // namespace FlexFlow

#endif

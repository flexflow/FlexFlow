#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_GET_SUBGRAPH_INPUT_EDGES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_GET_SUBGRAPH_INPUT_EDGES_H

#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"

namespace FlexFlow {

std::unordered_set<OpenDataflowEdge>
    get_subgraph_incoming_edges(OpenDataflowGraphView const &,
                                std::unordered_set<Node> const &);

} // namespace FlexFlow

#endif

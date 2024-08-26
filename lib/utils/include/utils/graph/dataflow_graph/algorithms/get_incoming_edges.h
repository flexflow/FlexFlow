#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_GET_INCOMING_EDGES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_GET_INCOMING_EDGES_H

#include "utils/graph/dataflow_graph/dataflow_graph_view.h"

namespace FlexFlow {

std::vector<DataflowEdge> get_incoming_edges(DataflowGraphView const &,
                                             Node const &);
std::unordered_set<DataflowEdge>
    get_incoming_edges(DataflowGraphView const &,
                       std::unordered_set<Node> const &);

} // namespace FlexFlow

#endif

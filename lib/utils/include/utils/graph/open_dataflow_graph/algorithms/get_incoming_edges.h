#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_GET_INCOMING_EDGES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_GET_INCOMING_EDGES_H

#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"

namespace FlexFlow {

std::unordered_set<DataflowInputEdge> get_incoming_edges(OpenDataflowGraphView const &);
std::vector<OpenDataflowEdge> get_incoming_edges(OpenDataflowGraphView const &,
                                                 Node const &);
std::unordered_map<Node, std::vector<OpenDataflowEdge>>
    get_incoming_edges(OpenDataflowGraphView const &,
                       std::unordered_set<Node> const &);


} // namespace FlexFlow

#endif

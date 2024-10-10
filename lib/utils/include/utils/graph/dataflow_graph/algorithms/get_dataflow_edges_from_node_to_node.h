#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_GET_DATAFLOW_EDGES_FROM_NODE_TO_NODE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_GET_DATAFLOW_EDGES_FROM_NODE_TO_NODE_H

#include "utils/graph/dataflow_graph/dataflow_graph_view.h"

namespace FlexFlow {

std::unordered_set<DataflowEdge> get_dataflow_edges_from_node_to_node(
    DataflowGraphView const &g, Node const &src, Node const &dst);

} // namespace FlexFlow

#endif

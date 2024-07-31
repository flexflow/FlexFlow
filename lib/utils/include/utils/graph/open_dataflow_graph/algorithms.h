#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_H

#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_value.dtg.h"

namespace FlexFlow {

std::unordered_set<OpenDataflowEdge> get_edges(OpenDataflowGraphView const &);
std::unordered_set<DataflowGraphInput>
    get_inputs(OpenDataflowGraphView const &);
std::vector<OpenDataflowValue> get_inputs(OpenDataflowGraphView const &,
                                          Node const &);
std::vector<OpenDataflowEdge> get_incoming_edges(OpenDataflowGraphView const &,
                                                 Node const &);
std::unordered_map<Node, std::vector<OpenDataflowEdge>>
    get_incoming_edges(OpenDataflowGraphView const &,
                       std::unordered_set<Node> const &);
std::unordered_set<OpenDataflowValue>
    get_open_dataflow_values(OpenDataflowGraphView const &);

} // namespace FlexFlow

#endif

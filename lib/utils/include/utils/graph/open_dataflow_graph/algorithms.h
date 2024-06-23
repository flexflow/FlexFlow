#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_H

#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"

namespace FlexFlow {

std::unordered_set<OpenDataflowEdge> get_edges(OpenDataflowGraphView const &);
std::vector<DataflowGraphInput> get_inputs(OpenDataflowGraphView const &);
std::vector<OpenDataflowEdge> get_incoming_edges(OpenDataflowGraphView const &);

} // namespace FlexFlow

#endif

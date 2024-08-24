#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_AS_DOT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_AS_DOT_H

#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"

namespace FlexFlow {

std::string as_dot(OpenDataflowGraphView const &);
std::string as_dot(OpenDataflowGraphView const &,
                   std::function<std::string(Node const &)> const &get_node_label, 
                   std::function<std::string(DataflowGraphInput const &)> const &get_input_label);

} // namespace FlexFlow

#endif

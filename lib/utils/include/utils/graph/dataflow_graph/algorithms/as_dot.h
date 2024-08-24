#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_AS_DOT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_AS_DOT_H

#include "utils/dot_file.h"
#include "utils/graph/dataflow_graph/dataflow_graph_view.h"

namespace FlexFlow {

std::string as_dot(DataflowGraphView const &);
void as_dot(DotFile<std::string> &, 
            DataflowGraphView const &,
            std::function<std::string(Node const &)> const &get_node_label);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_GET_SUBGRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_GET_SUBGRAPH_H

#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_value.dtg.h"
namespace FlexFlow {

OpenDataflowGraphView get_subgraph(OpenDataflowGraphView const &,
                                   std::unordered_set<Node> const &,
                                   std::vector<OpenDataflowValue> const &);

} // namespace FlexFlow

#endif

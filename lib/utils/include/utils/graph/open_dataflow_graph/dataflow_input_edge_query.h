#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_DATAFLOW_INPUT_EDGE_QUERY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_DATAFLOW_INPUT_EDGE_QUERY_H

#include "utils/graph/open_dataflow_graph/dataflow_input_edge.dtg.h"
#include "utils/graph/open_dataflow_graph/dataflow_input_edge_query.dtg.h"

namespace FlexFlow {

DataflowInputEdgeQuery dataflow_input_edge_query_all();
DataflowInputEdgeQuery dataflow_input_edge_query_none();
bool dataflow_input_edge_query_includes(DataflowInputEdgeQuery const &,
                                        DataflowInputEdge const &);

} // namespace FlexFlow

#endif

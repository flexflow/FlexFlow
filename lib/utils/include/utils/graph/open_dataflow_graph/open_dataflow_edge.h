#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_OPEN_DATAFLOW_EDGE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_OPEN_DATAFLOW_EDGE_H

#include "utils/graph/open_dataflow_graph/open_dataflow_edge.dtg.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_value.dtg.h"

namespace FlexFlow {

Node get_open_dataflow_edge_dst_node(OpenDataflowEdge const &);
int get_open_dataflow_edge_dst_idx(OpenDataflowEdge const &);
DataflowInput get_open_dataflow_edge_dst(OpenDataflowEdge const &);
OpenDataflowValue get_open_dataflow_edge_source(OpenDataflowEdge const &);
OpenDataflowEdge
    open_dataflow_edge_from_src_and_dst(OpenDataflowValue const &src,
                                        DataflowInput const &dst);

} // namespace FlexFlow

#endif

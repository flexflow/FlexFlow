#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_OPEN_DATAFLOW_EDGE_QUERY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_OPEN_DATAFLOW_EDGE_QUERY_H

#include "utils/graph/open_dataflow_graph/open_dataflow_edge.dtg.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge_query.dtg.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_value.dtg.h"

namespace FlexFlow {

OpenDataflowEdgeQuery open_dataflow_edge_query_all();
OpenDataflowEdgeQuery open_dataflow_edge_query_none();
bool open_dataflow_edge_query_includes(OpenDataflowEdgeQuery const &q,
                                       OpenDataflowEdge const &);
OpenDataflowEdgeQuery open_dataflow_edge_query_all_outgoing_from(OpenDataflowValue const &);
OpenDataflowEdgeQuery open_dataflow_edge_query_all_incoming_to(DataflowInput const &);
std::unordered_set<OpenDataflowEdge> apply_open_dataflow_edge_query(OpenDataflowEdgeQuery const &, std::unordered_set<OpenDataflowEdge> const &);

} // namespace FlexFlow

#endif

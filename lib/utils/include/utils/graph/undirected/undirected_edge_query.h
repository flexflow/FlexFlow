#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_UNDIRECTED_EDGE_QUERY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_UNDIRECTED_EDGE_QUERY_H

#include "utils/graph/undirected/undirected_edge_query.dtg.h"

namespace FlexFlow {

UndirectedEdgeQuery undirected_edge_query_all();

UndirectedEdgeQuery query_intersection(UndirectedEdgeQuery const &,
                                       UndirectedEdgeQuery const &);

} // namespace FlexFlow

#endif

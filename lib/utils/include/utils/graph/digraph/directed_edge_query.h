#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIRECTED_GRAPH_DIRECTED_EDGE_QUERY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIRECTED_GRAPH_DIRECTED_EDGE_QUERY_H

#include "utils/graph/digraph/directed_edge.dtg.h"
#include "utils/graph/digraph/directed_edge_query.dtg.h"

namespace FlexFlow {

DirectedEdgeQuery directed_edge_query_all();
bool matches_edge(DirectedEdgeQuery const &, DirectedEdge const &);
DirectedEdgeQuery query_intersection(DirectedEdgeQuery const &,
                                     DirectedEdgeQuery const &);

} // namespace FlexFlow

#endif

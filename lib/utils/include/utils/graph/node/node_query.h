#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_NODE_QUERY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_NODE_QUERY_H

#include "utils/graph/node/node_query.dtg.h"

namespace FlexFlow {

NodeQuery node_query_all();
NodeQuery query_intersection(NodeQuery const &, NodeQuery const &);
NodeQuery query_union(NodeQuery const &, NodeQuery const &);
std::unordered_set<Node> apply_node_query(NodeQuery const &, std::unordered_set<Node> const &);

} // namespace FlexFlow

#endif

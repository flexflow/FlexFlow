#include "utils/graph/node/node_query.h"

namespace FlexFlow {

NodeQuery node_query_all() {
  return NodeQuery{matchall<Node>()};
}

NodeQuery query_intersection(NodeQuery const &lhs, NodeQuery const &rhs) {

  std::unordered_set<Node> nodes;

  if (is_matchall(lhs.nodes) && !is_matchall(rhs.nodes)) {
    nodes = allowed_values(rhs.nodes);
  } else if (!is_matchall(lhs.nodes) && is_matchall(rhs.nodes)) {
    nodes = allowed_values(lhs.nodes);
  } else if (!is_matchall(lhs.nodes) && !is_matchall(rhs.nodes)) {
    nodes = allowed_values(query_intersection(lhs.nodes, rhs.nodes));
  }

  NodeQuery intersection_result = node_query_all();
  intersection_result.nodes = nodes;

  return intersection_result;
}

NodeQuery query_union(NodeQuery const &lhs, NodeQuery const &rhs) {
  NOT_IMPLEMENTED();
}

std::unordered_set<Node> apply_node_query(NodeQuery const &query, std::unordered_set<Node> const &ns) {
  return apply_query(query.nodes, ns);
}

} // namespace FlexFlow

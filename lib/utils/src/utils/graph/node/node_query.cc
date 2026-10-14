#include "utils/graph/node/node_query.h"
#include "utils/containers/set_of.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

NodeQuery node_query_all() {
  return NodeQuery{matchall<Node>()};
}

NodeQuery query_intersection(NodeQuery const &lhs, NodeQuery const &rhs) {

  std::set<Node> nodes;

  if (is_matchall(lhs.nodes) && !is_matchall(rhs.nodes)) {
    nodes = allowed_values(rhs.nodes);
  } else if (!is_matchall(lhs.nodes) && is_matchall(rhs.nodes)) {
    nodes = allowed_values(lhs.nodes);
  } else if (!is_matchall(lhs.nodes) && !is_matchall(rhs.nodes)) {
    nodes = allowed_values(query_intersection(lhs.nodes, rhs.nodes));
  }

  return NodeQuery{
      query_set<Node>::match_values_in(set_of(nodes)),
  };
}

NodeQuery query_union(NodeQuery const &lhs, NodeQuery const &rhs) {
  NOT_IMPLEMENTED();
}

std::set<Node> apply_node_query(NodeQuery const &query,
                                std::set<Node> const &ns) {
  return apply_query(query.nodes, ns);
}

} // namespace FlexFlow

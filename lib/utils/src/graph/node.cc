#include "utils/graph/node.h"
#include <sstream>
#include "utils/containers.h"

namespace FlexFlow {

NodeQuery::NodeQuery(std::unordered_set<Node> const &nodes)
    : NodeQuery(tl::optional<std::unordered_set<Node>>{nodes}) {}

NodeQuery::NodeQuery(tl::optional<std::unordered_set<Node>> const &nodes)
    : nodes(nodes) {}

NodeQuery query_intersection(NodeQuery const & lhs, NodeQuery const & rhs){
  if (!lhs.nodes.has_value()) {
    return rhs;
  } else if (!rhs.nodes.has_value()) {
    return lhs;
  } else {
    assert (lhs.nodes.has_value() && rhs.nodes.has_value());
    return { intersection(*lhs.nodes, *rhs.nodes) };
  }

}

} // namespace FlexFlow

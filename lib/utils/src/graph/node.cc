#include "utils/graph/node.h"
#include <sstream>

namespace FlexFlow {

std::ostream &operator<<(std::ostream &os, Node const &node) {
  return os << fmt::format("Node({})", node.value());
}

NodeQuery::NodeQuery(std::unordered_set<Node> const &nodes)
    : NodeQuery(tl::optional<std::unordered_set<Node>>{nodes}) {}

NodeQuery::NodeQuery(tl::optional<std::unordered_set<Node>> const &nodes)
    : nodes(nodes) {}

NodeQuery query_intersection(NodeQuery const &lhs, NodeQuery const &rhs) {
  assert(lhs != tl::nullopt && rhs != tl::nullopt);
  return intersection(*lhs.nodes, *rhs.nodes);
}

std::unordered_set<Node> GraphView::query_nodes(NodeQuery const &g) const {
  return this->ptr->query_nodes(g);
}

GraphView GraphView::unsafe_create(IGraphView const &graphView) {
  std::shared_ptr<IGraphView const> ptr((&graphView),
                                        [](IGraphView const *) {});
  return GraphView(ptr);
}

} // namespace FlexFlow

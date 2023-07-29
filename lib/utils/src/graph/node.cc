#include "utils/graph/node.h"
#include <sstream>

namespace FlexFlow {

std::ostream &operator<<(std::ostream &os, Node const &node) {
  return os << fmt::format("Node({})", node.value());
}

std::ostream &operator<<(std::ostream &os,
                         std::unordered_set<Node> const &nodes) {
  os << fmt::format("{ ");
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    if (it != nodes.begin()) {
      os << fmt::format(", ");
    }
    os << *it;
  }
  os << fmt::format(" }");
  return os;
}

NodeQuery NodeQuery::all() {
  return {matchall<Node>()};
}

NodeQuery query_intersection(NodeQuery const &lhs, NodeQuery const &rhs) {
  assert(lhs != nullopt && rhs != nullopt);
  std::unordered_set<Node> nodes =
      intersection(allowed_values(lhs.nodes), allowed_values(rhs.nodes));

  NodeQuery intersection_result = NodeQuery::all();
  intersection_result.nodes = nodes;

  return intersection_result;
}

std::unordered_set<Node> GraphView::query_nodes(NodeQuery const &g) const {
  return this->ptr->query_nodes(g);
}

/* unsafe_create:
1 create the std::shared_ptr<IGraphView const> ptr, and define a empty lambda function to delete the ptr.
2 use this ptr to create GraphView. It is read-only and it is not responsible for ownership management.
*/
GraphView GraphView::unsafe_create(IGraphView const &graphView) {
  std::shared_ptr<IGraphView const> ptr((&graphView),
                                        [](IGraphView const *) {});
  return GraphView(ptr);
}

} // namespace FlexFlow

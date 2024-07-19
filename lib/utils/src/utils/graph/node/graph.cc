#include "utils/graph/node/graph.h"

namespace FlexFlow {

Node Graph::add_node() {
  return get_ptr().add_node();
}

void Graph::add_node_unsafe(Node const &node) {
  get_ptr().add_node_unsafe(node);
}

void Graph::remove_node_unsafe(Node const &node) {
  get_ptr().remove_node_unsafe(node);
}

std::unordered_set<Node> Graph::query_nodes(NodeQuery const &q) const {
  return get_ptr().query_nodes(q);
}

IGraph const &Graph::get_ptr() const {
  return *std::dynamic_pointer_cast<IGraph const>(GraphView::ptr.get());
}

IGraph &Graph::get_ptr() {
  return *std::dynamic_pointer_cast<IGraph>(GraphView::ptr.get_mutable());
}

} // namespace FlexFlow

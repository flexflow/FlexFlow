#include "utils/graph/node.h"
#include "utils/graph/cow_ptr_t.h"
#include <sstream>

namespace FlexFlow {

NodeQuery NodeQuery::all() {
  return {matchall<Node>()};
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

  NodeQuery intersection_result = NodeQuery::all();
  intersection_result.nodes = nodes;

  return intersection_result;
}

std::unordered_set<Node> GraphView::query_nodes(NodeQuery const &g) const {
  return this->ptr->query_nodes(g);
}

bool is_ptr_equal(GraphView const &lhs, GraphView const &rhs) {
  return lhs.ptr == rhs.ptr;
}

GraphView::GraphView(cow_ptr_t<IGraphView> ptr) : ptr(ptr) {}

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

IGraph& Graph::get_ptr() const {
  return *std::reinterpret_pointer_cast<IGraph>(GraphView::ptr.get_mutable());
}

} // namespace FlexFlow

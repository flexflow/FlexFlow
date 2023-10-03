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

void swap(GraphView &lhs, GraphView &rhs) {
  std::swap(lhs.ptr, rhs.ptr);
}

std::unordered_set<Node> GraphView::query_nodes(NodeQuery const &g) const {
  return this->ptr->query_nodes(g);
}

GraphView::GraphView(std::shared_ptr<IGraphView const> ptr) : ptr(ptr) {}

void swap(Graph &lhs, Graph &rhs) {
  std::swap(lhs.ptr, rhs.ptr);
}

Node Graph::add_node() {
  get_ptr()->add_node();
}

Node Graph::add_node_unsafe() {
  get_ptr()->add_node_unsafe();
}

Node Graph::remove_node_unsafe() {
  get_ptr()->remove_node_unsafe();
}

std::unordered_set<Node> Graph::query_nodes(NodeQuery const &q) const {
  return get_ptr()->query_nodes(q);
}

Graph::Graph(cow_ptr_t<IGraph> _ptr) : ptr(_ptr) {
  assert(this->ptr.get() != nullptr);
}

cow_ptr_t<IGraph> Graph::get_ptr() const {
  return static_cast<cow_ptr_t<IGraph>>(ptr);
}

} // namespace FlexFlow

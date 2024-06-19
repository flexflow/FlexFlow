#include "utils/graph/undirected/undirected_graph.h"

namespace FlexFlow {

Node UndirectedGraph::add_node() {
  return this->get_ptr().add_node();
}

void UndirectedGraph::add_node_unsafe(Node const &n) {
  return this->get_ptr().add_node_unsafe(n);
}

void UndirectedGraph::remove_node_unsafe(Node const &n) {
  return this->get_ptr().remove_node_unsafe(n);
}

void UndirectedGraph::add_edge(UndirectedEdge const &e) {
  return this->get_ptr().add_edge(e);
}

void UndirectedGraph::remove_edge(UndirectedEdge const &e) {
  return this->get_ptr().remove_edge(e);
}

IUndirectedGraph const &UndirectedGraph::get_ptr() const {
  return *std::dynamic_pointer_cast<IUndirectedGraph const>(
      GraphView::ptr.get());
}

IUndirectedGraph &UndirectedGraph::get_ptr() {
  return *std::dynamic_pointer_cast<IUndirectedGraph>(
      GraphView::ptr.get_mutable());
}

std::unordered_set<UndirectedEdge>
    UndirectedGraph::query_edges(UndirectedEdgeQuery const &q) const {
  return this->get_ptr().query_edges(q);
}

std::unordered_set<Node>
    UndirectedGraph::query_nodes(NodeQuery const &q) const {
  return this->get_ptr().query_nodes(q);
}

} // namespace FlexFlow

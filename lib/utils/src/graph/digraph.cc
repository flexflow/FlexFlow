#include "utils/graph/digraph.h"

namespace FlexFlow {

void swap(DiGraph &lhs, DiGraph &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

Node DiGraph::add_node() {
  return this->ptr.get_mutable()->add_node();
}

void DiGraph::add_node_unsafe(Node const &n) {
  return this->ptr.get_mutable()->add_node_unsafe(n);
}

void DiGraph::remove_node_unsafe(Node const &n) {
  return this->ptr.get_mutable()->remove_node_unsafe(n);
}

void DiGraph::add_edge(DirectedEdge const &e) {
  return this->ptr.get_mutable()->add_edge(e);
}

void DiGraph::remove_edge(DirectedEdge const &e) {
  return this->ptr.get_mutable()->remove_edge(e);
}

std::unordered_set<DirectedEdge>
    DiGraph::query_edges(DirectedEdgeQuery const &q) const {
  return this->ptr->query_edges(q);
}

DiGraph::DiGraph(std::unique_ptr<IDiGraph> _ptr) : ptr(std::move(_ptr)) {}

} // namespace FlexFlow
#include "utils/graph/undirected.h"
#include "utils/containers.h"
#include <cassert>

namespace FlexFlow {

UndirectedEdgeQuery query_intersection(UndirectedEdgeQuery const &lhs,
                                       UndirectedEdgeQuery const &rhs) {
  return {
      query_intersection(lhs.nodes, rhs.nodes),
  };
}

void swap(UndirectedGraph &lhs, UndirectedGraph &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

Node UndirectedGraph::add_node() {
  return this->ptr.get_mutable()->add_node();
}

void UndirectedGraph::add_node_unsafe(Node const &n) {
  return this->ptr.get_mutable()->add_node_unsafe(n);
}

void UndirectedGraph::remove_node_unsafe(Node const &n) {
  return this->ptr.get_mutable()->remove_node_unsafe(n);
}

void UndirectedGraph::add_edge(UndirectedEdge const &e) {
  return this->ptr.get_mutable()->add_edge(e);
}

void UndirectedGraph::remove_edge(UndirectedEdge const &e) {
  return this->ptr.get_mutable()->remove_edge(e);
}

std::unordered_set<UndirectedEdge>
    UndirectedGraph::query_edges(UndirectedEdgeQuery const &q) const {
  return this->ptr->query_edges(q);
}

UndirectedGraph::UndirectedGraph(std::unique_ptr<IUndirectedGraph> _ptr)
    : ptr(std::move(_ptr)) {}

} // namespace FlexFlow

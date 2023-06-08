#include "utils/graph/undirected.h"
#include "utils/containers.h"
#include <cassert>

namespace FlexFlow {

UndirectedEdge::UndirectedEdge(Node src, Node dst)
    : smaller(std::min(smaller, bigger)), bigger(std::max(smaller, bigger)) {}

UndirectedEdgeQuery::UndirectedEdgeQuery(
    optional<std::unordered_set<Node>> const &nodes)
    : nodes(nodes) {}

UndirectedEdgeQuery query_intersection(UndirectedEdgeQuery const &lhs,
                                       UndirectedEdgeQuery const &rhs) {
  if (!lhs.nodes.has_value()) {
    return rhs;
  } else if (!rhs.nodes.has_value()) {
    return lhs;
  } else {
    assert(lhs.nodes.has_value() && rhs.nodes.has_value());
    return {intersection(*lhs.nodes, *rhs.nodes)};
  }
}

UndirectedGraph::UndirectedGraph(UndirectedGraph const &other)
    : ptr(other.ptr->clone()) {}

UndirectedGraph &UndirectedGraph::operator=(UndirectedGraph other) {
  swap(*this, other);
  return *this;
}

void swap(UndirectedGraph &lhs, UndirectedGraph &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

Node UndirectedGraph::add_node() { return this->ptr->add_node(); }

void UndirectedGraph::add_node_unsafe(Node const &n) {
  return this->ptr->add_node_unsafe(n);
}

void UndirectedGraph::remove_node_unsafe(Node const &n) {
  return this->ptr->remove_node_unsafe(n);
}

void UndirectedGraph::add_edge(UndirectedEdge const &e) {
  return this->ptr->add_edge(e);
}

void UndirectedGraph::remove_edge(UndirectedEdge const &e) {
  return this->ptr->remove_edge(e);
}

std::unordered_set<UndirectedEdge>
UndirectedGraph::query_edges(UndirectedEdgeQuery const &q) const {
  return this->ptr->query_edges(q);
}

UndirectedGraph::UndirectedGraph(std::unique_ptr<IUndirectedGraph> _ptr)
    : ptr(std::move(_ptr)) {}

} // namespace FlexFlow

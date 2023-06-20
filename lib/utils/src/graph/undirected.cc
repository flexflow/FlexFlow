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
    : ptr(other.ptr.get_mutable()) {}

UndirectedGraph &UndirectedGraph::operator=(UndirectedGraph other) {
  swap(*this, other);
  return *this;
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
UndirectedGraph:: operator UndirectedGraphView() const {
    return UndirectedGraphView(std::shared_ptr<IUndirectedGraphView const>(ptr.get_mutable()));
}

std::unordered_set<UndirectedEdge>
    UndirectedGraph::query_edges(UndirectedEdgeQuery const &q) const {
  return this->ptr.get_mutable()->query_edges(q);
}

UndirectedGraph::UndirectedGraph(std::shared_ptr<IUndirectedGraph> ptr)
    : ptr(ptr) {}
  
std::unordered_set<UndirectedEdge> UndirectedGraphView::query_edges(UndirectedEdgeQuery const& g)  const {
  return this->ptr->query_edges(g);
}

} // namespace FlexFlow

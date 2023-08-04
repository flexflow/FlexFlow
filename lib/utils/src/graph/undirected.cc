#include "utils/graph/undirected.h"
#include "utils/containers.h"
#include <cassert>

namespace FlexFlow {

UndirectedEdge::UndirectedEdge(Node const &src, Node const &dst)
    : smaller(std::min(smaller, bigger)), bigger(std::max(smaller, bigger)) {}

UndirectedEdgeQuery UndirectedEdgeQuery::all() {
  return {matchall<Node>()};
}

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

UndirectedGraph::operator UndirectedGraphView() const {
  return UndirectedGraphView::unsafe_create_without_ownership(*this->ptr.get());
}

std::unordered_set<UndirectedEdge>
    UndirectedGraphView::query_edges(UndirectedEdgeQuery const &q) const {
  return this->ptr->query_edges(q);
}

std::unordered_set<Node>
    UndirectedGraphView::query_nodes(NodeQuery const &q) const {
  return this->ptr->query_nodes(q);
}

/* unsafe_create_without_ownership:
1 create the std::shared_ptr<IUndirectedGraphView const> ptr, and define a empty
lambda function to delete the ptr. 2 use this ptr to create UndirectedGraphView.
It is read-only and it is not responsible for ownership management.
*/
UndirectedGraphView
    UndirectedGraphView::unsafe_create_without_ownership(IUndirectedGraphView const &g) {
  std::shared_ptr<IUndirectedGraphView const> ptr(
      (&g), [](IUndirectedGraphView const *) {});
  return UndirectedGraphView(ptr);
}

UndirectedGraphView::operator GraphView() const {
  return GraphView(this->ptr, should_only_be_used_internally_tag_t{});
}

} // namespace FlexFlow

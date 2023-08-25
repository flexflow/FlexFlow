#include "utils/graph/undirected.h"
#include "utils/graph/internal.h"
#include "utils/containers.h"
#include "utils/graph/node.h"
#include <cassert>

namespace FlexFlow {

UndirectedEdge::UndirectedEdge(Node const &n1, Node const &n2)
    : smaller(std::min(n1, n2)), bigger(std::max(n1, n2)) {}

bool is_connected_to(UndirectedEdge const &e, Node const &n) {
  return e.bigger == n || e.smaller == n;
}

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

std::unordered_set<Node>
    UndirectedGraph::query_nodes(NodeQuery const &q) const {
  return this->ptr->query_nodes(q);
}

UndirectedGraph::UndirectedGraph(cow_ptr_t<IUndirectedGraph> _ptr)
    : ptr(std::move(_ptr)) {}

UndirectedGraph::operator UndirectedGraphView() const {
  return GraphInternal::create_undirectedgraphview(this->ptr.get());
}

std::unordered_set<UndirectedEdge>
    UndirectedGraphView::query_edges(UndirectedEdgeQuery const &q) const {
  return this->ptr->query_edges(q);
}

std::unordered_set<Node>
    UndirectedGraphView::query_nodes(NodeQuery const &q) const {
  return this->ptr->query_nodes(q);
}

// Set the shared_ptr's destructor to a nop so that effectively there is no
// ownership
UndirectedGraphView UndirectedGraphView::unsafe_create_without_ownership(
    IUndirectedGraphView const &g) {
  std::shared_ptr<IUndirectedGraphView const> ptr(
      (&g), [](IUndirectedGraphView const *) {});
  return UndirectedGraphView(ptr);
}

UndirectedGraphView::operator GraphView() const {
  return GraphInternal::create_graphview(this->ptr);
}

} // namespace FlexFlow

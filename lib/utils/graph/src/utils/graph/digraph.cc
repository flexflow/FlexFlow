#include "utils/graph/digraph.h"
#include "utils/containers.h"
#include "utils/graph/digraph_interfaces.h"
#include "utils/graph/internal.h"

namespace FlexFlow {

void swap(DiGraphView &lhs, DiGraphView &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

bool is_ptr_equal(DiGraphView const &lhs, DiGraphView const &rhs) {
  return lhs.ptr == rhs.ptr;
}

std::unordered_set<Node> DiGraphView::query_nodes(NodeQuery const &q) const {
  return this->get_ptr()->query_nodes(q);
}

std::unordered_set<DirectedEdge>
    DiGraphView::query_edges(EdgeQuery const &query) const {
  return get_ptr()->query_edges(query);
}

DiGraphView::DiGraphView(cow_ptr_t<IDiGraphView> ptr) : GraphView(ptr) {}

cow_ptr_t<IDiGraphView> IDiGraphView::get_ptr() const {
  return static_cast<cow_ptr_t<IDiGraphView>>(ptr);
}

void swap(DiGraph &lhs, DiGraph &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

Node DiGraph::add_node() {
  return this->get_ptr().get_mutable()->add_node();
}

void DiGraph::add_node_unsafe(Node const &n) {
  return this->get_ptr().get_mutable()->add_node_unsafe(n);
}

void DiGraph::remove_node_unsafe(Node const &n) {
  return this->get_ptr().get_mutable()->remove_node_unsafe(n);
}

void DiGraph::add_edge(DirectedEdge const &e) {
  return this->get_ptr().get_mutable()->add_edge(e);
}

void DiGraph::remove_edge(DirectedEdge const &e) {
  return this->get_ptr().get_mutable()->remove_edge(e);
}

std::unordered_set<Node> DiGraph::query_nodes(NodeQuery const &q) const {
  return this->get_ptr()->query_nodes(q);
}

std::unordered_set<DirectedEdge>
    DiGraph::query_edges(DirectedEdgeQuery const &q) const {
  return this->get_ptr()->query_edges(q);
}

DiGraph::DiGraph(cow_ptr_t<IDiGraph> _ptr) : DiGraphView(_ptr) {}

} // namespace FlexFlow

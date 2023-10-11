#include "utils/graph/digraph.h"
#include "utils/containers.h"
#include "utils/graph/digraph_interfaces.h"
#include "utils/graph/internal.h"

namespace FlexFlow {

bool is_ptr_equal(DiGraphView const &lhs, DiGraphView const &rhs) {
  return lhs.GraphView::ptr == rhs.GraphView::ptr;
}

std::unordered_set<Node> DiGraphView::query_nodes(NodeQuery const &q) const {
  return this->get_ptr()->query_nodes(q);
}

std::unordered_set<DirectedEdge>
    DiGraphView::query_edges(EdgeQuery const &query) const {
  return get_ptr()->query_edges(query);
}

// DiGraphView::DiGraphView(cow_ptr_t<IDiGraphView> ptr) : GraphView(ptr) {}

cow_ptr_t<IDiGraphView> DiGraphView::get_ptr() const {
  return cow_ptr_t(
      std::dynamic_pointer_cast<IDiGraphView>(GraphView::ptr.get_mutable()));
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

// DiGraph::DiGraph(cow_ptr_t<IDiGraph> _ptr) : DiGraphView(_ptr) {}

} // namespace FlexFlow

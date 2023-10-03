#include "utils/graph/multidigraph.h"
#include "utils/graph/internal.h"
#include "utils/graph/multidigraph_interfaces.h"

namespace FlexFlow {

void swap(MultiDiGraphView &lhs, MultiDiGraphView &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

std::unordered_set<Node>
    MultiDiGraphView::query_nodes(NodeQuery const &q) const {
  return this->get_ptr()->query_nodes(q);
}

std::unordered_set<MultiDiEdge>
    MultiDiGraphView::query_edges(MultiDiEdgeQuery const &q) const {
  return this->get_ptr()->query_edges(q);
}

MultiDiGraphView::MultiDiGraphView(cow_ptr_t<IMultiDiGraphView> ptr)
    : DiGraphView(ptr) {}

cow_ptr_t<IMultiDiGraphView> MultiDiGraphView::get_ptr() const {
  return static_cast<cow_ptr_t<IMultiDiGraphView>>(ptr);
}

void swap(MultiDiGraph &lhs, MultiDiGraph &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

Node MultiDiGraph::add_node() {
  return this->get_ptr().get_mutable()->add_node();
}

NodePort MultiDiGraph::add_node_port() {
  return this->get_ptr().get_mutable()->add_node_port();
}

void MultiDiGraph::add_node_port_unsafe(NodePort const &np) {
  return this->get_ptr().get_mutable()->add_node_port_unsafe(np);
}

void MultiDiGraph::add_node_unsafe(Node const &n) {
  return this->get_ptr().get_mutable()->add_node_unsafe(n);
}

void MultiDiGraph::remove_node_unsafe(Node const &n) {
  return this->get_ptr().get_mutable()->remove_node_unsafe(n);
}

void MultiDiGraph::add_edge(MultiDiEdge const &e) {
  return this->get_ptr().get_mutable()->add_edge(e);
}

void MultiDiGraph::remove_edge(MultiDiEdge const &e) {
  return this->get_ptr().get_mutable()->remove_edge(e);
}

std::unordered_set<MultiDiEdge>
    MultiDiGraph::query_edges(MultiDiEdgeQuery const &q) const {
  return this->get_ptr()->query_edges(q);
}

std::unordered_set<Node> MultiDiGraph::query_nodes(NodeQuery const &q) const {
  return this->get_ptr()->query_nodes(q);
}

MultiDiGraph::MultiDiGraph(cow_ptr_t<IMultiDiGraph> ptr)
    : MultiDiGraphView(ptr) {}

cow_ptr_t<IMultiDiGraph> MultiDiGraph::get_ptr() const {
  return static_cast<cow_ptr_t<IMultiDiGraph>>(ptr);
}

} // namespace FlexFlow

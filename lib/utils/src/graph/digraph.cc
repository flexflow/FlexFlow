#include "utils/graph/digraph.h"
#include "utils/containers.h"
#include "utils/graph/digraph_interfaces.h"

namespace FlexFlow {

std::unordered_set<Node> DiGraphView::query_nodes(NodeQuery const &q) const {
  return this->get_ptr().query_nodes(q);
}

std::unordered_set<DirectedEdge>
    DiGraphView::query_edges(EdgeQuery const &query) const {
  return get_ptr().query_edges(query);
}

IDiGraphView const &DiGraphView::get_ptr() const {
  return *std::dynamic_pointer_cast<IDiGraphView const>(
      GraphView::ptr.get());
}

Node DiGraph::add_node() {
  return this->get_ptr().add_node();
}

void DiGraph::add_node_unsafe(Node const &n) {
  return this->get_ptr().add_node_unsafe(n);
}

void DiGraph::remove_node_unsafe(Node const &n) {
  return this->get_ptr().remove_node_unsafe(n);
}

void DiGraph::add_edge(DirectedEdge const &e) {
  return this->get_ptr().add_edge(e);
}

void DiGraph::remove_edge(DirectedEdge const &e) {
  return this->get_ptr().remove_edge(e);
}

std::unordered_set<Node> DiGraph::query_nodes(NodeQuery const &q) const {
  return this->get_ptr().query_nodes(q);
}

std::unordered_set<DirectedEdge>
    DiGraph::query_edges(DirectedEdgeQuery const &q) const {
  return this->get_ptr().query_edges(q);
}

IDiGraph &DiGraph::get_ptr() {
  return *std::dynamic_pointer_cast<IDiGraph>(GraphView::ptr.get_mutable());
}

IDiGraph const &DiGraph::get_ptr() const {
  return *std::dynamic_pointer_cast<IDiGraph const>(
      GraphView::ptr.get_mutable());
}
} // namespace FlexFlow

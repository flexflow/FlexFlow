#include "utils/graph/multidigraph/multidigraph.h"

namespace FlexFlow {

Node MultiDiGraph::add_node() {
  return this->get_interface().add_node();
}

MultiDiEdge MultiDiGraph::add_edge(Node const &src, Node const &dst) {
  return this->get_interface().add_edge(src, dst);
}

void MultiDiGraph::remove_node(Node const &n) {
  this->get_interface().remove_node(n);
}

void MultiDiGraph::remove_edge(MultiDiEdge const &e) {
  this->get_interface().remove_edge(e);
}

std::unordered_set<Node> MultiDiGraph::query_nodes(NodeQuery const &q) const {
  return this->get_interface().query_nodes(q);
}

std::unordered_set<MultiDiEdge> MultiDiGraph::query_edges(MultiDiEdgeQuery const &q) const {
  return this->get_interface().query_edges(q);
}

Node MultiDiGraph::get_multidiedge_src(MultiDiEdge const &e) const {
  return this->get_interface().get_multidiedge_src(e);
}

Node MultiDiGraph::get_multidiedge_dst(MultiDiEdge const &e) const {
  return this->get_interface().get_multidiedge_dst(e);
}

IMultiDiGraph &MultiDiGraph::get_interface() {
  return *std::dynamic_pointer_cast<IMultiDiGraph>(
      GraphView::ptr.get_mutable());
}

IMultiDiGraph const &MultiDiGraph::get_interface() const {
  return *std::dynamic_pointer_cast<IMultiDiGraph const>(GraphView::ptr.get());
}

} // namespace FlexFlow

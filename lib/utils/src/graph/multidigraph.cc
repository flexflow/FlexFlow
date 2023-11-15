#include "utils/graph/multidigraph.h"
#include "utils/graph/multidigraph_interfaces.h"

namespace FlexFlow {

std::unordered_set<DirectedEdge>
    IMultiDiGraphView::query_edges(DirectedEdgeQuery const &q) const {
  return transform(
      query_edges(MultiDiEdgeQuery{
          q.srcs, q.dsts, matchall<NodePort>(), matchall<NodePort>()}),
      [](MultiDiEdge const &e) {
        return DirectedEdge{e.src, e.dst};
      });
}

std::unordered_set<Node>
    MultiDiGraphView::query_nodes(NodeQuery const &q) const {
  return this->get_ptr().query_nodes(q);
}

std::unordered_set<MultiDiEdge>
    MultiDiGraphView::query_edges(MultiDiEdgeQuery const &q) const {
  return this->get_ptr().query_edges(q);
}

IMultiDiGraphView const &MultiDiGraphView::get_ptr() const {
  return *std::dynamic_pointer_cast<IMultiDiGraphView const>(GraphView::ptr.get());
}

Node MultiDiGraph::add_node() {
  return this->get_ptr().add_node();
}

NodePort MultiDiGraph::add_node_port() {
  return this->get_ptr().add_node_port();
}

void MultiDiGraph::add_node_port_unsafe(NodePort const &np) {
  return this->get_ptr().add_node_port_unsafe(np);
}

void MultiDiGraph::add_node_unsafe(Node const &n) {
  return this->get_ptr().add_node_unsafe(n);
}

void MultiDiGraph::remove_node_unsafe(Node const &n) {
  return this->get_ptr().remove_node_unsafe(n);
}

void MultiDiGraph::add_edge(MultiDiEdge const &e) {
  return this->get_ptr().add_edge(e);
}

void MultiDiGraph::remove_edge(MultiDiEdge const &e) {
  return this->get_ptr().remove_edge(e);
}

std::unordered_set<MultiDiEdge>
    MultiDiGraph::query_edges(MultiDiEdgeQuery const &q) const {
  return this->get_ptr().query_edges(q);
}

std::unordered_set<Node> MultiDiGraph::query_nodes(NodeQuery const &q) const {
  return this->get_ptr().query_nodes(q);
}

IMultiDiGraph &MultiDiGraph::get_ptr() const {
  return *std::dynamic_pointer_cast<IMultiDiGraph>(
      GraphView::ptr.get_mutable());
}

} // namespace FlexFlow

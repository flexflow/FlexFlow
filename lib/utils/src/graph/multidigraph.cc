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
  return this->get_ptr()->query_nodes(q);
}

std::unordered_set<MultiDiEdge>
    MultiDiGraphView::query_edges(MultiDiEdgeQuery const &q) const {
  std::cout<<"this->get_ptr()->query_edges(q).size():"<<this->get_ptr()->query_edges(q).size()<<std::endl;
  return this->get_ptr()->query_edges(q);
}

cow_ptr_t<IMultiDiGraphView> MultiDiGraphView::get_ptr() const {
  return cow_ptr_t(std::reinterpret_pointer_cast<IMultiDiGraphView>(
      GraphView::ptr.get_mutable()));
}

Node MultiDiGraph::add_node() {
 //return this->get_ptr().get_mutable()->add_node();
return this->get_ptr().get1()->add_node();
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
  std::cout<<" MultiDiGraph::query_edges:"<<std::endl;
  return this->get_ptr()->query_edges(q);
}

std::unordered_set<Node> MultiDiGraph::query_nodes(NodeQuery const &q) const {
  return this->get_ptr()->query_nodes(q);
}

cow_ptr_t<IMultiDiGraph> MultiDiGraph::get_ptr() const {
  return cow_ptr_t(std::reinterpret_pointer_cast<IMultiDiGraph>(
      GraphView::ptr.get_mutable()));
}

} // namespace FlexFlow

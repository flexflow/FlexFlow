#include "utils/graph/undirected.h"
#include "utils/containers.h"
#include "utils/graph/internal.h"
#include "utils/graph/node.h"
#include <cassert>

namespace FlexFlow {

std::unordered_set<Node> UndirectedGraphView::query_nodes(NodeQuery const &q) const {
  return get_ptr()->query_nodes(q);
}

std::unordered_set<UndirectedEdge> UndirectedGraphView::query_edges(UndirectedEdgeQuery const &q) const {
  return get_ptr()->query_edges(q);
}

UndirectedGraphView UndirectedGraphView(cow_ptr_t<IUndirectedGraphView const> ptr) : GraphView(ptr) {}

Node UndirectedGraph::add_node() {
  return this->get_ptr().get_mutable()->add_node();
}

void UndirectedGraph::add_node_unsafe(Node const &n) {
  return this->get_ptr().get_mutable()->add_node_unsafe(n);
}

void UndirectedGraph::remove_node_unsafe(Node const &n) {
  return this->get_ptr().get_mutable()->remove_node_unsafe(n);
}

void UndirectedGraph::add_edge(UndirectedEdge const &e) {
  return this->get_ptr().get_mutable()->add_edge(e);
}

void UndirectedGraph::remove_edge(UndirectedEdge const &e) {
  return this->get_ptr().get_mutable()->remove_edge(e);
}

std::unordered_set<UndirectedEdge>
    UndirectedGraph::query_edges(UndirectedEdgeQuery const &q) const {
  return this->get_ptr()->query_edges(q);
}

std::unordered_set<Node>
    UndirectedGraph::query_nodes(NodeQuery const &q) const {
  return this->get_ptr()->query_nodes(q);
}

UndirectedGraph::UndirectedGraph(cow_ptr_t<IUndirectedGraph> _ptr)
  : UndirectedGraphView(_ptr) {}

std::unordered_set<UndirectedEdge>
    UndirectedGraphView::query_edges(UndirectedEdgeQuery const &q) const {
  return this->get_ptr()->query_edges(q);
}

std::unordered_set<Node>
    UndirectedGraphView::query_nodes(NodeQuery const &q) const {
  return this->get_ptr()->query_nodes(q);
}

cow_ptr_t<IUndirectedGraph> UndirectedGraphView::get_ptr() const {
  return static_cast<cow_ptr_t<IUndirectedGraph>>(ptr);
}

} // namespace FlexFlow

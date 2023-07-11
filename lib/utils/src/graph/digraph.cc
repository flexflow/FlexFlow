#include "utils/graph/digraph.h"
#include <optional>

namespace FlexFlow {

std::ostream &operator<<(std::ostream &s, DirectedEdge const &e) {
  return (s << "DirectedEdge{" << e.src.value() << " -> " << e.dst.value()
            << "}");
}

DirectedEdgeQuery::DirectedEdgeQuery(
    optional<std::unordered_set<Node>> const &srcs,
    optional<std::unordered_set<Node>> const &dsts)
    : srcs(srcs), dsts(dsts) {}

void swap(DiGraph &lhs, DiGraph &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

Node DiGraph::add_node() {
  return this->ptr.get_mutable()->add_node();
}

void DiGraph::add_node_unsafe(Node const &n) {
  return this->ptr.get_mutable()->add_node_unsafe(n);
}

void DiGraph::remove_node_unsafe(Node const &n) {
  return this->ptr.get_mutable()->remove_node_unsafe(n);
}

void DiGraph::add_edge(DirectedEdge const &e) {
  return this->ptr.get_mutable()->add_edge(e);
}

void DiGraph::remove_edge(DirectedEdge const &e) {
  return this->ptr.get_mutable()->remove_edge(e);
}

std::unordered_set<DirectedEdge>
    DiGraph::query_edges(DirectedEdgeQuery const &q) const {
  return this->ptr->query_edges(q);
}

DiGraph::DiGraph(std::unique_ptr<IDiGraph> _ptr) : ptr(std::move(_ptr)) {}

DiGraphView::operator GraphView() const {
  return GraphView(this->ptr);
}

bool DiGraphView::operator==(DiGraphView const &other) const {
  return ptr == other.ptr;
}

bool DiGraphView::operator!=(DiGraphView const &other) const {
  return ptr != other.ptr;
}

std::unordered_set<Node> DiGraphView::query_nodes(NodeQuery const& q) const {
  return this->ptr->query_nodes(q);
}

std::unordered_set<DirectedEdge> DiGraphView::query_edges(EdgeQuery const & query) const {
  return ptr->query_edges(query);
}

DiGraphView unsafe_create(IDiGraphView const &graphView) {
  std::shared_ptr<IDiGraphView const> ptr((&graphView),
  [](IDiGraphView const *){});
  /*
  1 use the graphView to creae the std::shared_ptr<IDiGraphView const> ptr, and define a empty lambda function to delete the ptr
  2 we use this ptr to create a DiGraphView, this DiGraphView is read-only. It creates a DiGraphView object that is not responsible for ownership management
  */
  return DiGraphView(ptr);
}

DirectedEdgeQuery query_intersection(DirectedEdgeQuery const &lhs, DirectedEdgeQuery const &rhs){
  assert(lhs != tl::nullopt);
  assert(rhs != tl::nullopt);
  assert (lhs.srcs.has_value() && lhs.dsts.has_value() && rhs.srcs.has_value() && rhs.dsts.has_value());

  tl::optional<std::unordered_set<Node>> srcs_t1 = intersection(*lhs.srcs, *rhs.srcs);
  tl::optional<std::unordered_set<Node>> dsts_t1 = intersection(*lhs.dsts, *rhs.dsts);

  return DirectedEdgeQuery(srcs_t1, dsts_t1);
}



} // namespace FlexFlow

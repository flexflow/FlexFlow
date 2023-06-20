#include "utils/graph/digraph.h"
#include "utils/containers.h"

namespace FlexFlow {

DirectedEdge::DirectedEdge(Node src, Node dst) : src(src), dst(dst) {}

std::ostream &operator<<(std::ostream &s, DirectedEdge const &e) {
  return (s << "DirectedEdge{" << e.src.value() << " -> " << e.dst.value()
            << "}");
}

DirectedEdgeQuery::DirectedEdgeQuery(
    tl::optional<std::unordered_set<Node>> const &srcs,
    tl::optional<std::unordered_set<Node>> const &dsts)
    : srcs(srcs), dsts(dsts) {}

DirectedEdgeQuery query_intersection(DirectedEdgeQuery const &lhs, DirectedEdgeQuery const &rhs){
  assert (lhs.srcs.has_value() && lhs.dsts.has_value() && rhs.srcs.has_value() && rhs.dsts.has_value());

  tl::optional<std::unordered_set<Node>> srcs_t1 = intersection(*lhs.srcs, *rhs.srcs);
  tl::optional<std::unordered_set<Node>> dsts_t1 = intersection(*lhs.dsts, *rhs.dsts);

  return DirectedEdgeQuery(srcs_t1, dsts_t1);
}

// DiGraph::DiGraph(DiGraph const &other) : ptr(other->ptr.get_mutable()->clone()) {} //TODO

DiGraph &DiGraph::operator=(DiGraph other) {
  swap(*this, other);
  return *this;
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
  return this->ptr.get_mutable()->query_edges(q);
}

DiGraph::DiGraph(std::shared_ptr<IDiGraph> ptr) : ptr(std::move(ptr)) {}



} // namespace FlexFlow

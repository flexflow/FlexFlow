#include "utils/graph/digraph.h"
#include "utils/containers.h"

namespace FlexFlow {

std::ostream &operator<<(std::ostream &s, DirectedEdge const &e) {
  std::string str = fmt::format("DirectedEdge(src={}, dst={})", e.src, e.dst);
  return s << str;
}

void swap(DiGraph &lhs, DiGraph &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

bool is_ptr_equal(DiGraphView const &lhs, DiGraphView const &rhs) {
  return lhs.ptr == rhs.ptr;
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
  return GraphView(this->ptr, should_only_be_used_internally_tag_t{});
}

std::unordered_set<Node> DiGraphView::query_nodes(NodeQuery const &q) const {
  return this->ptr->query_nodes(q);
}

std::unordered_set<DirectedEdge>
    DiGraphView::query_edges(EdgeQuery const &query) const {
  return ptr->query_edges(query);
}

// Set the shared_ptr's destructor to a nop so that effectively there is no
// ownership
DiGraphView DiGraphView::unsafe_create_without_ownership(
    IDiGraphView const &graphView) {
  std::shared_ptr<IDiGraphView const> ptr((&graphView),
                                          [](IDiGraphView const *) {});
  return DiGraphView(ptr);
}

DirectedEdgeQuery DirectedEdgeQuery::all() {
  return {matchall<Node>(), matchall<Node>()};
}

DirectedEdgeQuery query_intersection(DirectedEdgeQuery const &lhs,
                                     DirectedEdgeQuery const &rhs) {
  std::unordered_set<Node> srcs_tl;
  if (is_matchall(lhs.srcs) && !is_matchall(rhs.srcs)) {
    srcs_tl = allowed_values(rhs.srcs);
  } else if (!is_matchall(lhs.srcs) && is_matchall(rhs.srcs)) {
    srcs_tl = allowed_values(lhs.srcs);
  } else if (!is_matchall(lhs.srcs) && !is_matchall(rhs.srcs)) {
    srcs_tl = allowed_values(query_intersection(lhs.srcs, rhs.srcs));
  }

  std::unordered_set<Node> dsts_tl;
  if (is_matchall(lhs.dsts) && !is_matchall(rhs.dsts)) {
    dsts_tl = allowed_values(rhs.dsts);
  } else if (!is_matchall(lhs.dsts) && is_matchall(rhs.dsts)) {
    dsts_tl = allowed_values(lhs.dsts);
  } else if (!is_matchall(lhs.dsts) && !is_matchall(rhs.dsts)) {
    dsts_tl = allowed_values(query_intersection(lhs.dsts, rhs.dsts));
  }

  DirectedEdgeQuery result = DirectedEdgeQuery::all();
  result.srcs = srcs_tl;
  result.dsts = dsts_tl;
  return result;
}

} // namespace FlexFlow

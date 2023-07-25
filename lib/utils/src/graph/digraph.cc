#include "utils/graph/digraph.h"
#include <regex>

namespace FlexFlow {

std::ostream &operator<<(std::ostream &s, DirectedEdge const &e) {
  std::string str = fmt::format("DirectedEdge(src={}, dst={})", e.src, e.dst);
  return s << str;
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
  return this->ptr->query_edges(q);
}

DiGraph::DiGraph(std::unique_ptr<IDiGraph> _ptr) : ptr(std::move(_ptr)) {}

DiGraphView::operator GraphView() const {
  return GraphView::unsafe_create(*(this->ptr.get()));
}

bool DiGraphView::operator==(DiGraphView const &other) const {
  return ptr == other.ptr;
}

bool DiGraphView::operator!=(DiGraphView const &other) const {
  return ptr != other.ptr;
}

std::unordered_set<Node> DiGraphView::query_nodes(NodeQuery const &q) const {
  return this->ptr->query_nodes(q);
}

std::unordered_set<DirectedEdge>
    DiGraphView::query_edges(EdgeQuery const &query) const {
  return ptr->query_edges(query);
}

/* unsafe_create:
1 use the graphView to creae the std::shared_ptr<IDiGraphView const> ptr, and
define a empty lambda function to delete the ptr 2 we use this ptr to create a
DiGraphView, this DiGraphView is read-only. It creates a DiGraphView object that
is not responsible for ownership management
*/
DiGraphView DiGraphView::unsafe_create(IDiGraphView const &graphView) {
  std::shared_ptr<IDiGraphView const> ptr((&graphView),
                                          [](IDiGraphView const *) {});
  return DiGraphView(ptr);
}

DirectedEdgeQuery DirectedEdgeQuery::all() {
  return {matchall<Node>(), matchall<Node>()};
}

DirectedEdgeQuery query_intersection(DirectedEdgeQuery const &lhs,
                                     DirectedEdgeQuery const &rhs) {
  assert(lhs != tl::nullopt);
  assert(rhs != tl::nullopt);
  assert(lhs.srcs.has_value() && lhs.dsts.has_value() && rhs.srcs.has_value() &&
         rhs.dsts.has_value());

  std::unordered_set<Node> srcs_t1 =
      intersection(allowed_values(lhs.srcs), allowed_values(rhs.srcs));
  std::unordered_set<Node> dsts_t1 =
      intersection(allowed_values(lhs.dsts), allowed_values(rhs.dsts));

  DirectedEdgeQuery result = DirectedEdgeQuery::all();
  result.srcs = srcs_t1;
  result.dsts = dsts_t1;
  return result;
}

} // namespace FlexFlow

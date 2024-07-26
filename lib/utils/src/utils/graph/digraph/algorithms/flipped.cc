#include "utils/graph/digraph/algorithms/flipped.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

FlippedView::FlippedView(DiGraphView const &g) : g(g) {}

std::unordered_set<DirectedEdge>
    FlippedView::query_edges(DirectedEdgeQuery const &query) const {
  std::unordered_set<DirectedEdge> result =
      this->g.query_edges(DirectedEdgeQuery{query.dsts, query.srcs});
  return transform(
      result, [](DirectedEdge const &e) { return flipped_directed_edge(e); });
}

std::unordered_set<Node>
    FlippedView::query_nodes(NodeQuery const &query) const {
  return this->g.query_nodes(query);
}

FlippedView *FlippedView::clone() const {
  return new FlippedView(g);
}

DiGraphView flipped(DiGraphView const &g) {
  return DiGraphView::create<FlippedView>(g);
}

DirectedEdge flipped_directed_edge(DirectedEdge const &e) {
  return DirectedEdge{e.dst, e.src};
}

} // namespace FlexFlow

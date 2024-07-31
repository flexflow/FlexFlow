#include "utils/graph/digraph/algorithms/transitive_reduction.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/materialize_digraph_view.h"
#include "utils/graph/instances/adjacency_digraph.h"

namespace FlexFlow {

DirectedEdgeMaskView::DirectedEdgeMaskView(
    DiGraphView const &g, std::unordered_set<DirectedEdge> const &edge_mask)
    : g(g), edge_mask(edge_mask) {}

std::unordered_set<DirectedEdge>
    DirectedEdgeMaskView::query_edges(DirectedEdgeQuery const &q) const {
  return intersection(g.query_edges(q), this->edge_mask);
}

std::unordered_set<Node>
    DirectedEdgeMaskView::query_nodes(NodeQuery const &q) const {
  return g.query_nodes(q);
}

DirectedEdgeMaskView *DirectedEdgeMaskView::clone() const {
  return new DirectedEdgeMaskView(this->g, this->edge_mask);
}

DiGraphView transitive_reduction(DiGraphView const &g) {
  std::unordered_set<DirectedEdge> edge_mask = get_edges(g);

  while (true) {
    std::unordered_set<DirectedEdge> new_edge_mask = edge_mask;
    for (DirectedEdge const &e1 : edge_mask) {
      for (DirectedEdge const &e2 : edge_mask) {
        if (e1.dst == e2.src && e1 != e2) {
          DirectedEdge trans_edge = DirectedEdge{e1.src, e2.dst};
          if (contains(new_edge_mask, trans_edge)) {
            new_edge_mask.erase(trans_edge);
          }
        }
      }
    }

    if (new_edge_mask == edge_mask) {
      break;
    } else {
      edge_mask = new_edge_mask;
    }
  }

  return DiGraphView::create<DirectedEdgeMaskView>(g, edge_mask);
}

} // namespace FlexFlow

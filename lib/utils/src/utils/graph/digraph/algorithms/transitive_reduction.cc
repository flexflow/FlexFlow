#include "utils/graph/digraph/algorithms/transitive_reduction.h"
#include "utils/containers/contains.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_descendants.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/graph/digraph/algorithms/materialize_digraph_view.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"

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

DiGraphView unchecked_transitive_reduction(DiGraphView const &g) {
  std::unordered_set<DirectedEdge> edge_mask = get_edges(g);

  for (Node const &n1 : get_nodes(g)) {
    std::unordered_set<Node> n1_descendants = get_descendants(g, n1);
    for (Node const &n2 : get_nodes(g)) {
      if (contains(n1_descendants, n2)) {
        std::unordered_set<Node> n2_descendants = get_descendants(g, n2);
        for (Node const &n3 : get_nodes(g)) {
          if (contains(n2_descendants, n3)) {
            edge_mask.erase(DirectedEdge{n1, n3});
          }
        }
      }
    }
  }
  return DiGraphView::create<DirectedEdgeMaskView>(g, edge_mask);
}

DiGraphView transitive_reduction(DiGraphView const &g) {
  assert(is_acyclic(g));
  return unchecked_transitive_reduction(g);
}

} // namespace FlexFlow

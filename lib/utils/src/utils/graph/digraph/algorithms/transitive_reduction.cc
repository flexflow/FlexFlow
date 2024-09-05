#include "utils/graph/digraph/algorithms/transitive_reduction.h"
#include "utils/bidict/algorithms/bidict_from_enumerating.h"
#include "utils/containers/vector_of.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/materialize_digraph_view.h"
#include "utils/graph/digraph/algorithms/transitive_closure.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/containers/is_subseteq_of.h"

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
  // Logic dropped down to raw adjacency matrix for performance.
  // The version going through the full graph abstraction was 
  // incredibly slow (> minutes) for even moderately sized graphs 
  // (i.e., 200 nodes) without optimization enabled.
  //
  // transitive_closure inlined to avoid any drifts in node numbering
  // between transitive_closure and transitive_reduction
  
  bidict<int, Node> nodes = bidict_from_enumerating(get_nodes(g));
  int num_nodes = nodes.size();

  std::vector<bool> edge_matrix(num_nodes * num_nodes, false);

  auto has_edge = [&](int src_idx, int dst_idx) -> std::vector<bool>::reference {
    return edge_matrix[src_idx * num_nodes + dst_idx];
  };

  for (DirectedEdge const &e : get_edges(g)) {
    has_edge(nodes.at_r(e.src), nodes.at_r(e.dst)) = true;
  }

  // compute transitive closure
  // see https://cs.winona.edu/lin/cs440/ch08-2.pdf slide 8-8
  for (int k = 0; k < num_nodes; k++) {
    for (int i = 0; i < num_nodes; i++) {
      if (has_edge(i, k)) {
        for (int j = 0; j < num_nodes; j++) {
          if (has_edge(k, j)) {
            has_edge(i, j) = true;
          }
        }
      }
    }
  }

  DiGraph result = materialize_digraph_view<AdjacencyDiGraph>(g);
  // compute transitive reduction
  // see https://stackoverflow.com/a/6702198
  std::unordered_set<DirectedEdge> edge_mask = get_edges(g);
  for (int j = 0; j < num_nodes; j++) {
    for (int i = 0; i < num_nodes; i++) {
      if (has_edge(i, j)) {
        for (int k = 0; k < num_nodes; k++) {
          if (has_edge(j, k)) {
            has_edge(i, k) = false;
            result.remove_edge(DirectedEdge{nodes.at_l(i), nodes.at_l(k)});
          }
        }
      }
    }
  }

  return result;
}

} // namespace FlexFlow

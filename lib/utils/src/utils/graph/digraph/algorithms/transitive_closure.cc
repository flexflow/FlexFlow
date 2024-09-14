#include "utils/graph/digraph/algorithms/transitive_closure.h"
#include "utils/bidict/algorithms/bidict_from_enumerating.h"
#include "utils/containers/vector_of.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/digraph_has_edge.h"
#include "utils/graph/digraph/algorithms/materialize_digraph_view.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

DiGraphView transitive_closure(DiGraphView const &g) {
  // Logic dropped down to raw adjacency matrix for performance.
  // The version going through the full graph abstraction was
  // incredibly slow (> minutes) for even moderately sized graphs
  // (i.e., 200 nodes) without optimization enabled.

  bidict<int, Node> nodes = bidict_from_enumerating(get_nodes(g));
  std::unordered_set<DirectedEdge> edges = get_edges(g);

  int num_nodes = nodes.size();

  std::vector<bool> edge_matrix(num_nodes * num_nodes, false);

  auto has_edge = [&](int src_idx,
                      int dst_idx) -> std::vector<bool>::reference {
    return edge_matrix[src_idx * num_nodes + dst_idx];
  };

  for (DirectedEdge const &e : get_edges(g)) {
    has_edge(nodes.at_r(e.src), nodes.at_r(e.dst)) = true;
  }

  DiGraph result = materialize_digraph_view<AdjacencyDiGraph>(g);
  for (int k = 0; k < num_nodes; k++) {
    for (int i = 0; i < num_nodes; i++) {
      if (has_edge(i, k)) {
        for (int j = 0; j < num_nodes; j++) {
          if (has_edge(k, j)) {
            has_edge(i, j) = true;
            result.add_edge(DirectedEdge{nodes.at_l(i), nodes.at_l(j)});
          }
        }
      }
    }
  }

  return result;
}

} // namespace FlexFlow

#include "utils/graph/digraph/algorithms/complete_bipartite_composite/is_complete_bipartite_digraph.h"
#include "utils/containers/get_first.h"
#include "utils/containers/set_minus.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

bool is_complete_bipartite_digraph(DiGraphView const &g) {
  return is_complete_bipartite_digraph(g, get_sources(g));
}

bool is_complete_bipartite_digraph(DiGraphView const &g,
                                   std::unordered_set<Node> const &srcs) {
  std::unordered_set<Node> sinks = set_minus(get_nodes(g), srcs);

  std::unordered_set<DirectedEdge> edges = get_edges(g);

  std::unordered_set<DirectedEdge> expected_edges;
  for (Node const &src : srcs) {
    for (Node const &sink : sinks) {
      expected_edges.insert(DirectedEdge{src, sink});
    }
  }

  return edges == expected_edges;
}

} // namespace FlexFlow

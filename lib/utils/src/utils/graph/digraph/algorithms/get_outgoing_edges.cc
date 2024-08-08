#include "utils/graph/digraph/algorithms/get_outgoing_edges.h"
#include "utils/containers/group_by.h"

namespace FlexFlow {

std::unordered_map<Node, std::unordered_set<DirectedEdge>>
    get_outgoing_edges(DiGraphView const &g,
                       std::unordered_set<Node> const &ns) {
  std::unordered_map<Node, std::unordered_set<DirectedEdge>> result =
      group_by(g.query_edges(DirectedEdgeQuery{
                   query_set<Node>{ns},
                   query_set<Node>::matchall(),
               }),
               [](DirectedEdge const &e) { return e.src; });

  for (Node const &n : ns) {
    result[n];
  }

  return result;
}

std::unordered_set<DirectedEdge> get_outgoing_edges(DiGraphView const &g,
                                                    Node const &n) {
  return g.query_edges(DirectedEdgeQuery{
      query_set<Node>{n},
      query_set<Node>::matchall(),
  });
}

} // namespace FlexFlow

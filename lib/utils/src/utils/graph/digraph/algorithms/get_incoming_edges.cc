#include "utils/graph/digraph/algorithms/get_incoming_edges.h"
#include "utils/containers/group_by.h"

namespace FlexFlow {

std::unordered_set<DirectedEdge> get_incoming_edges(DiGraphView const &g,
                                                    Node const &n) {
  return g.query_edges(DirectedEdgeQuery{
      query_set<Node>::matchall(),
      query_set<Node>{n},
  });
}

std::unordered_map<Node, std::unordered_set<DirectedEdge>>
    get_incoming_edges(DiGraphView const &g,
                       std::unordered_set<Node> const &ns) {
  std::unordered_map<Node, std::unordered_set<DirectedEdge>> result =
      group_by(g.query_edges(DirectedEdgeQuery{
                   query_set<Node>::matchall(),
                   query_set<Node>{ns},
               }),
               [](DirectedEdge const &e) { return e.dst; });

  for (Node const &n : ns) {
    result[n];
  }

  return result;
}

} // namespace FlexFlow

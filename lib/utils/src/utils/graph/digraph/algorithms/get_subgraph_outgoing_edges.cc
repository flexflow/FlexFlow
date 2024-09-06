#include "utils/graph/digraph/algorithms/get_subgraph_outgoing_edges.h"
#include "utils/containers/set_minus.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

std::unordered_set<DirectedEdge> get_subgraph_outgoing_edges(DiGraphView const &g,
                                                             std::unordered_set<Node> const &subgraph_nodes) {
  std::unordered_set<Node> external_nodes = set_minus(get_nodes(g), subgraph_nodes);
  DirectedEdgeQuery query = DirectedEdgeQuery{
    query_set<Node>{subgraph_nodes}, 
    query_set<Node>{external_nodes}
  };
  return g.query_edges(query);
}

} // namespace FlexFlow

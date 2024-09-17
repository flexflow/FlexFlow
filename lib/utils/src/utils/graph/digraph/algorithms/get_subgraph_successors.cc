#include "utils/graph/digraph/algorithms/get_subgraph_successors.h"
#include "utils/graph/digraph/algorithms/get_subgraph_outgoing_edges.h"

namespace FlexFlow {

std::unordered_set<Node>
    get_subgraph_successors(DiGraphView const &g,
                            std::unordered_set<Node> const &subgraph_nodes) {
  std::unordered_set<Node> successors =
      transform(get_subgraph_outgoing_edges(g, subgraph_nodes),
                [](DirectedEdge const &e) { return e.dst; });

  return successors;
}

} // namespace FlexFlow

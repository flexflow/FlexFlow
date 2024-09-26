#include "utils/graph/digraph/algorithms/get_edges_from_subgraph_to_subgraph.h"
#include "utils/containers/are_disjoint.h"

namespace FlexFlow {

std::unordered_set<DirectedEdge> get_edges_from_subgraph_to_subgraph(DiGraphView const &g,
                                                                     std::unordered_set<Node> const &src_subgraph,
                                                                     std::unordered_set<Node> const &dst_subgraph) {
  if (!are_disjoint(src_subgraph, dst_subgraph)) {
    throw mk_runtime_error(fmt::format("get_edges_from_subgraph_to_subgraph(DiGraphView, ...) expected src_subgraph and dst_subgraph to be disjoint, "
                                       "but found src_subgraph={}, dst_subgraph={}", src_subgraph, dst_subgraph));
  }

  return g.query_edges(DirectedEdgeQuery{
    /*srcs=*/query_set<Node>{src_subgraph},
    /*dsts=*/query_set<Node>{dst_subgraph},
  });
}

} // namespace FlexFlow

#include "utils/graph/digraph/algorithms/get_edges_from_subgraph_to_subgraph.h"
#include "utils/containers/are_disjoint.h"
#include "utils/containers/set_of.h"
#include "utils/exception.h"

namespace FlexFlow {

std::set<DirectedEdge>
    get_edges_from_subgraph_to_subgraph(DiGraphView const &g,
                                        std::set<Node> const &src_subgraph,
                                        std::set<Node> const &dst_subgraph) {
  if (!are_disjoint(src_subgraph, dst_subgraph)) {
    PANIC(fmt::format("get_edges_from_subgraph_to_subgraph(DiGraphView, ...) "
                      "expected src_subgraph and dst_subgraph to be disjoint, "
                      "but found src_subgraph={}, dst_subgraph={}",
                      src_subgraph,
                      dst_subgraph));
  }

  return g.query_edges(DirectedEdgeQuery{
      /*srcs=*/query_set<Node>::match_values_in(set_of(src_subgraph)),
      /*dsts=*/query_set<Node>::match_values_in(set_of(dst_subgraph)),
  });
}

} // namespace FlexFlow

#include "utils/graph/dataflow_graph/algorithms/get_outgoing_edges.h"
#include "utils/containers/set_minus.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

std::unordered_set<DataflowEdge>
    get_outgoing_edges(DataflowGraphView const &g,
                       std::unordered_set<Node> const &ns,
                       IncludeInternalEdges include_internal_edges) {
  query_set<Node> dst_query = [&] {
    if (include_internal_edges == IncludeInternalEdges::YES) {
      return query_set<Node>::matchall();
    } else {
      assert(include_internal_edges == IncludeInternalEdges::NO);

      std::unordered_set<Node> all_nodes = get_nodes(g);
      return query_set<Node>{set_minus(all_nodes, ns)};
    }
  }();

  DataflowEdgeQuery query = DataflowEdgeQuery{
      query_set<Node>{ns},
      query_set<int>::matchall(),
      dst_query,
      query_set<int>::matchall(),
  };

  return g.query_edges(query);
}

} // namespace FlexFlow

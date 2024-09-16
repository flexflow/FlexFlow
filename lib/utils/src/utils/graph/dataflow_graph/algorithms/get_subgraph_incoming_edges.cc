#include "utils/graph/dataflow_graph/algorithms/get_subgraph_incoming_edges.h"
#include "utils/containers/set_minus.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

std::unordered_set<DataflowEdge>
    get_subgraph_incoming_edges(DataflowGraphView const &g,
                                std::unordered_set<Node> const &ns) {

  std::unordered_set<Node> all_nodes = get_nodes(g);
  query_set<Node> src_query = query_set<Node>{set_minus(all_nodes, ns)};

  DataflowEdgeQuery query = DataflowEdgeQuery{
      src_query,
      query_set<int>::matchall(),
      query_set<Node>{ns},
      query_set<int>::matchall(),
  };

  return g.query_edges(query);
}

} // namespace FlexFlow

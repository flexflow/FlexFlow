#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph_input_edges.h"
#include "utils/containers/set_minus.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

std::unordered_set<OpenDataflowEdge>
    get_subgraph_incoming_edges(OpenDataflowGraphView const &g,
                                std::unordered_set<Node> const &ns) {
  std::unordered_set<Node> nodes_not_in_ns = set_minus(get_nodes(g), ns);

  OpenDataflowEdgeQuery query = OpenDataflowEdgeQuery{
      DataflowInputEdgeQuery{
          query_set<DataflowGraphInput>::matchall(),
          query_set<Node>{ns},
          query_set<int>::matchall(),
      },
      DataflowEdgeQuery{
          query_set<Node>{nodes_not_in_ns},
          query_set<int>::matchall(),
          query_set<Node>{ns},
          query_set<int>::matchall(),
      },
  };

  return g.query_edges(query);
}

} // namespace FlexFlow

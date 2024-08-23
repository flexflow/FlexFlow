#include "utils/graph/dataflow_graph/algorithms/get_outgoing_edges.h"

namespace FlexFlow {

std::unordered_set<DataflowEdge> get_outgoing_edges(DataflowGraphView const &g, std::unordered_set<Node> const &ns) {
  DataflowEdgeQuery query = DataflowEdgeQuery{
    query_set<Node>{ns},
    query_set<int>::matchall(),
    query_set<Node>::matchall(),
    query_set<int>::matchall(),
  };
  return g.query_edges(query);
}

} // namespace FlexFlow

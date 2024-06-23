#include "utils/graph/open_dataflow_graph/algorithms.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge_query.h"

namespace FlexFlow {

std::unordered_set<OpenDataflowEdge> get_edges(OpenDataflowGraphView const &g) {
  return g.query_edges(open_dataflow_edge_query_all());
}

std::vector<DataflowGraphInput> get_inputs(OpenDataflowGraphView const &g) {
  return g.get_inputs();
}

std::vector<OpenDataflowEdge> get_incoming_edges(OpenDataflowGraphView const &g, Node const &n) {
  return sorted_by(g.query_edges(OpenDataflowEdgeQuery{
    DataflowInputEdgeQuery{
      query_set<DataflowGraphInput>::matchall(),
      {n},
      query_set<int>::matchall(),
    },
    DataflowEdgeQuery{
      query_set<Node>::matchall(),
      query_set<int>::matchall(),
      {n},
      query_set<int>::matchall(),
    },
  }), [](OpenDataflowEdge const &l, OpenDataflowEdge const &r) { return get_open_dataflow_edge_dst_idx(l) < get_open_dataflow_edge_dst_idx(r); });
}

} // namespace FlexFlow

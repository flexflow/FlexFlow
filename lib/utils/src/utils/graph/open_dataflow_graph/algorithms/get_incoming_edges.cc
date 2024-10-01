#include "utils/graph/open_dataflow_graph/algorithms/get_incoming_edges.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/sorted_by.h"
#include "utils/containers/transform.h"
#include "utils/graph/dataflow_graph/dataflow_edge_query.h"
#include "utils/graph/open_dataflow_graph/dataflow_input_edge_query.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge.h"

namespace FlexFlow {

std::unordered_set<DataflowInputEdge>
    get_incoming_edges(OpenDataflowGraphView const &g) {
  std::unordered_set<OpenDataflowEdge> raw_edges =
      g.query_edges(OpenDataflowEdgeQuery{
          dataflow_input_edge_query_all(),
          dataflow_edge_query_none(),
      });

  return transform(raw_edges, [](OpenDataflowEdge const &e) {
    return e.get<DataflowInputEdge>();
  });
}

std::vector<OpenDataflowEdge> get_incoming_edges(OpenDataflowGraphView const &g,
                                                 Node const &n) {
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
                   }),
                   [](OpenDataflowEdge const &l, OpenDataflowEdge const &r) {
                     return get_open_dataflow_edge_dst_idx(l) <
                            get_open_dataflow_edge_dst_idx(r);
                   });
}

std::unordered_map<Node, std::vector<OpenDataflowEdge>>
    get_incoming_edges(OpenDataflowGraphView const &g,
                       std::unordered_set<Node> const &ns) {
  return generate_map(ns,
                      [&](Node const &n) { return get_incoming_edges(g, n); });
}

} // namespace FlexFlow

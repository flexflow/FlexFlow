#include "utils/graph/open_dataflow_graph/algorithms.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/group_by.h"
#include "utils/containers/without_order.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge_query.h"
#include "utils/containers/transform.h"
#include "utils/containers/sorted_by.h"

namespace FlexFlow {

std::unordered_set<OpenDataflowEdge> get_edges(OpenDataflowGraphView const &g) {
  return g.query_edges(open_dataflow_edge_query_all());
}

std::unordered_set<DataflowGraphInput>
    get_inputs(OpenDataflowGraphView const &g) {
  return g.get_inputs();
}

std::vector<OpenDataflowValue> get_inputs(OpenDataflowGraphView const &g,
                                          Node const &n) {
  return transform(get_incoming_edges(g, n), [](OpenDataflowEdge const &e) {
    return get_open_dataflow_edge_source(e);
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

std::unordered_set<OpenDataflowValue>
    get_open_dataflow_values(OpenDataflowGraphView const &g) {
  return set_union(
      transform(
          without_order(g.get_inputs()),
          [](DataflowGraphInput const &gi) { return OpenDataflowValue{gi}; }),
      transform(get_all_dataflow_outputs(g),
                [](DataflowOutput const &o) { return OpenDataflowValue{o}; }));
}

} // namespace FlexFlow

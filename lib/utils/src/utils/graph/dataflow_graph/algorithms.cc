#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/dataflow_graph/dataflow_edge_query.h"
#include "utils/graph/dataflow_graph/dataflow_output_query.h"
#include "utils/containers/transform.h"
#include "utils/containers/sorted_by.h"

namespace FlexFlow {

std::unordered_set<DataflowEdge> get_edges(DataflowGraphView const &g) {
  return g.query_edges(dataflow_edge_query_all());
}

std::vector<DataflowEdge> get_incoming_edges(DataflowGraphView const &g,
                                             Node const &n) {
  return sorted_by(g.query_edges(DataflowEdgeQuery{
                       query_set<Node>::matchall(),
                       query_set<int>::matchall(),
                       {n},
                       query_set<int>::matchall(),
                   }),
                   [](DataflowEdge const &l, DataflowEdge const &r) {
                     return l.dst.idx < r.dst.idx;
                   });
}

std::vector<DataflowOutput> get_inputs(DataflowGraphView const &g,
                                       Node const &n) {
  return transform(get_incoming_edges(g, n),
                   [](DataflowEdge const &e) { return e.src; });
}

std::vector<DataflowOutput> get_outputs(DataflowGraphView const &g,
                                        Node const &n) {
  return sorted_by(g.query_outputs(DataflowOutputQuery{
                       query_set<Node>{n},
                       query_set<int>::matchall(),
                   }),
                   [](DataflowOutput const &l, DataflowOutput const &r) {
                     return l.idx < r.idx;
                   });
}

std::unordered_set<DataflowOutput>
    get_all_dataflow_outputs(DataflowGraphView const &g) {
  return g.query_outputs(dataflow_output_query_all());
}

} // namespace FlexFlow

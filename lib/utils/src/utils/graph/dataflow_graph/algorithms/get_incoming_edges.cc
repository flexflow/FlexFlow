#include "utils/graph/dataflow_graph/algorithms/get_incoming_edges.h"
#include "utils/containers/sorted_by.h"

namespace FlexFlow {

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

std::unordered_set<DataflowEdge> get_incoming_edges(DataflowGraphView const &g, std::unordered_set<Node> const &ns) {
  DataflowEdgeQuery query = DataflowEdgeQuery{
    query_set<Node>::matchall(),
    query_set<int>::matchall(),
    query_set<Node>{ns},
    query_set<int>::matchall(),
  };
  return g.query_edges(query);
}

} // namespace FlexFlow

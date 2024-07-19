#include "utils/graph/dataflow_graph/dataflow_edge_query.h"

namespace FlexFlow {

DataflowEdgeQuery dataflow_edge_query_all() {
  return DataflowEdgeQuery{
      query_set<Node>::matchall(),
      query_set<int>::matchall(),
      query_set<Node>::matchall(),
      query_set<int>::matchall(),
  };
}

DataflowEdgeQuery dataflow_edge_query_none() {
  return DataflowEdgeQuery{
      query_set<Node>::match_none(),
      query_set<int>::match_none(),
      query_set<Node>::match_none(),
      query_set<int>::match_none(),
  };
}

bool dataflow_edge_query_includes_dataflow_edge(DataflowEdgeQuery const &q,
                                                DataflowEdge const &e) {
  return includes(q.src_nodes, e.src.node) &&
         includes(q.dst_nodes, e.dst.node) && includes(q.src_idxs, e.src.idx) &&
         includes(q.dst_idxs, e.dst.idx);
}

} // namespace FlexFlow

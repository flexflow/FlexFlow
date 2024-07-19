#include "utils/graph/open_dataflow_graph/dataflow_input_edge_query.h"

namespace FlexFlow {

DataflowInputEdgeQuery dataflow_input_edge_query_all() {
  return DataflowInputEdgeQuery{
      query_set<DataflowGraphInput>::matchall(),
      query_set<Node>::matchall(),
      query_set<int>::matchall(),
  };
}
DataflowInputEdgeQuery dataflow_input_edge_query_none() {
  return DataflowInputEdgeQuery{
      query_set<DataflowGraphInput>::match_none(),
      query_set<Node>::match_none(),
      query_set<int>::match_none(),
  };
}

bool dataflow_input_edge_query_includes(DataflowInputEdgeQuery const &q,
                                        DataflowInputEdge const &e) {
  return includes(q.srcs, e.src) && includes(q.dst_nodes, e.dst.node) &&
         includes(q.dst_idxs, e.dst.idx);
}

} // namespace FlexFlow

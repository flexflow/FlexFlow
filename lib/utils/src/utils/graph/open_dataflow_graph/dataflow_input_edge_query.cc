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

DataflowInputEdgeQuery
    dataflow_input_edge_query_for_edge(DataflowInputEdge const &e) {
  return DataflowInputEdgeQuery{
      query_set<DataflowGraphInput>{e.src},
      query_set<Node>{e.dst.node},
      query_set<int>{e.dst.idx},
  };
}

DataflowInputEdgeQuery
    dataflow_input_edge_query_all_outgoing_from(DataflowGraphInput const &src) {
  return DataflowInputEdgeQuery{
      query_set<DataflowGraphInput>{src},
      query_set<Node>::matchall(),
      query_set<int>::matchall(),
  };
}

DataflowInputEdgeQuery
    dataflow_input_edge_query_all_incoming_to(DataflowInput const &dst) {
  return DataflowInputEdgeQuery{
      query_set<DataflowGraphInput>::matchall(),
      query_set<Node>{dst.node},
      query_set<int>{dst.idx},
  };
}

} // namespace FlexFlow

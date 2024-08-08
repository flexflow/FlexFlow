#include "utils/graph/dataflow_graph/dataflow_output_query.h"

namespace FlexFlow {

DataflowOutputQuery dataflow_output_query_all() {
  return DataflowOutputQuery{
      query_set<Node>::matchall(),
      query_set<int>::matchall(),
  };
}

DataflowOutputQuery dataflow_output_query_none() {
  return DataflowOutputQuery{
      query_set<Node>::match_none(),
      query_set<int>::match_none(),
  };
}

bool dataflow_output_query_includes_dataflow_output(
    DataflowOutputQuery const &q, DataflowOutput const &o) {
  return includes(q.nodes, o.node) && includes(q.output_idxs, o.idx);
}

DataflowOutputQuery dataflow_output_query_for_output(DataflowOutput const &o) {
  return DataflowOutputQuery{
    query_set<Node>{o.node},
    query_set<int>{o.idx},
  };
}

std::unordered_set<DataflowOutput> apply_dataflow_output_query(DataflowOutputQuery const &q, std::unordered_set<DataflowOutput> const &os) {
  return filter(os, [&](DataflowOutput const &o) { return dataflow_output_query_includes_dataflow_output(q, o); });
}

} // namespace FlexFlow

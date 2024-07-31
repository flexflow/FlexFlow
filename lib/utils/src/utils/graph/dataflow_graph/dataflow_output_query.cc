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

} // namespace FlexFlow

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

} // namespace FlexFlow

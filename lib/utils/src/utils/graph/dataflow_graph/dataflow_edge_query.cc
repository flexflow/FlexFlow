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

} // namespace FlexFlow

#include "utils/graph/open_dataflow_graph/open_dataflow_edge_query.h"
#include "utils/graph/open_dataflow_graph/dataflow_input_edge_query.h"
#include "utils/graph/dataflow_graph/dataflow_edge_query.h"

namespace FlexFlow {

OpenDataflowEdgeQuery open_dataflow_edge_query_all() {
  return OpenDataflowEdgeQuery{
    dataflow_input_edge_query_all(),
    dataflow_edge_query_all(),
  };
}

OpenDataflowEdgeQuery open_dataflow_edge_query_none() {
  return OpenDataflowEdgeQuery{
    dataflow_input_edge_query_none(),
    dataflow_edge_query_none(),
  };
}

} // namespace FlexFlow

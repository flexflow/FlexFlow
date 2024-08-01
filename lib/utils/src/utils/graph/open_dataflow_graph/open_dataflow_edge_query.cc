#include "utils/graph/open_dataflow_graph/open_dataflow_edge_query.h"
#include "utils/graph/dataflow_graph/dataflow_edge_query.h"
#include "utils/graph/open_dataflow_graph/dataflow_input_edge_query.h"
#include "utils/overload.h"

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

bool open_dataflow_edge_query_includes(OpenDataflowEdgeQuery const &q,
                                       OpenDataflowEdge const &open_e) {
  return open_e.visit<bool>(overload{
      [&](DataflowEdge const &e) {
        return dataflow_edge_query_includes_dataflow_edge(q.standard_edge_query,
                                                          e);
      },
      [&](DataflowInputEdge const &e) {
        return dataflow_input_edge_query_includes(q.input_edge_query, e);
      },
  });
}

} // namespace FlexFlow

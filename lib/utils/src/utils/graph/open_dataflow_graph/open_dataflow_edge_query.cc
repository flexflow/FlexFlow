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

OpenDataflowEdgeQuery
    open_dataflow_edge_query_all_outgoing_from(OpenDataflowValue const &src) {
  return src.visit<OpenDataflowEdgeQuery>(overload{
      [](DataflowOutput const &o) {
        return OpenDataflowEdgeQuery{
            dataflow_input_edge_query_none(),
            dataflow_edge_query_all_outgoing_from(o),
        };
      },
      [](DataflowGraphInput const &i) {
        return OpenDataflowEdgeQuery{
            dataflow_input_edge_query_all_outgoing_from(i),
            dataflow_edge_query_none(),
        };
      },
  });
}

OpenDataflowEdgeQuery
    open_dataflow_edge_query_all_incoming_to(DataflowInput const &dst) {
  return OpenDataflowEdgeQuery{
      dataflow_input_edge_query_all_incoming_to(dst),
      dataflow_edge_query_all_incoming_to(dst),
  };
}

std::unordered_set<OpenDataflowEdge> apply_open_dataflow_edge_query(
    OpenDataflowEdgeQuery const &q,
    std::unordered_set<OpenDataflowEdge> const &es) {
  return filter(es, [&](OpenDataflowEdge const &e) {
    return open_dataflow_edge_query_includes(q, e);
  });
}

} // namespace FlexFlow

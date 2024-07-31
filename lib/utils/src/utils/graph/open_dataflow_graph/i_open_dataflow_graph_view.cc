#include "utils/graph/open_dataflow_graph/i_open_dataflow_graph_view.h"
#include "utils/containers/transform.h"
#include "utils/graph/open_dataflow_graph/dataflow_input_edge_query.h"

namespace FlexFlow {

std::unordered_set<DataflowEdge>
    IOpenDataflowGraphView::query_edges(DataflowEdgeQuery const &q) const {
  OpenDataflowEdgeQuery open_query = OpenDataflowEdgeQuery{
      dataflow_input_edge_query_none(),
      q,
  };

  std::unordered_set<OpenDataflowEdge> open_edges =
      this->query_edges(open_query);

  return transform(open_edges, [](OpenDataflowEdge const &e) {
    return e.get<DataflowEdge>();
  });
}

} // namespace FlexFlow

#include "utils/graph/dataflow_graph/i_dataflow_graph_view.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

std::unordered_set<DirectedEdge>
    IDataflowGraphView::query_edges(DirectedEdgeQuery const &q) const {
  DataflowEdgeQuery dataflow_query = DataflowEdgeQuery{
      q.srcs,
      matchall<int>(),
      q.dsts,
      matchall<int>(),
  };
  std::unordered_set<DataflowEdge> dataflow_edges =
      this->query_edges(dataflow_query);

  return transform(dataflow_edges, [](DataflowEdge const &e) {
    return DirectedEdge{e.src.node, e.dst.node};
  });
}

} // namespace FlexFlow

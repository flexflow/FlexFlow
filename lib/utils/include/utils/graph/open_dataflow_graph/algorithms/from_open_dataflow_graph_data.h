#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_FROM_OPEN_DATAFLOW_GRAPH_DATA_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_FROM_OPEN_DATAFLOW_GRAPH_DATA_H

#include "utils/graph/open_dataflow_graph/algorithms/open_dataflow_graph_data.dtg.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"

namespace FlexFlow {

struct FromOpenDataflowGraphDataView final : virtual public IOpenDataflowGraphView {
  FromOpenDataflowGraphDataView(OpenDataflowGraphData const &);  

  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
  std::unordered_set<OpenDataflowEdge> query_edges(OpenDataflowEdgeQuery const &) const override;
  std::unordered_set<DataflowOutput> query_outputs(DataflowOutputQuery const &) const override;
  std::unordered_set<DataflowGraphInput> get_inputs() const override;

  FromOpenDataflowGraphDataView *clone() const override;
private:
  OpenDataflowGraphData data;
};

OpenDataflowGraphView from_open_dataflow_graph_data(OpenDataflowGraphData const &);

} // namespace FlexFlow

#endif

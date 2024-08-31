#ifndef _FLEXFLOW_LIB_UTILS_SRC_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_VIEW_AS_OPEN_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_SRC_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_VIEW_AS_OPEN_DATAFLOW_GRAPH_H

#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"

namespace FlexFlow {

struct ViewDataflowGraphAsOpen final
  : public IOpenDataflowGraphView {
public:
  ViewDataflowGraphAsOpen() = delete;
  ViewDataflowGraphAsOpen(DataflowGraphView const &);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
  std::unordered_set<OpenDataflowEdge> query_edges(OpenDataflowEdgeQuery const &) const override;
  std::unordered_set<DataflowOutput> query_outputs(DataflowOutputQuery const &) const override;
  std::unordered_set<DataflowGraphInput> get_inputs() const override;

  ViewDataflowGraphAsOpen *clone() const override;

  ~ViewDataflowGraphAsOpen() = default;
private:
  DataflowGraphView g;
};

OpenDataflowGraphView view_as_open_dataflow_graph(DataflowGraphView const &);

} // namespace FlexFlow

#endif

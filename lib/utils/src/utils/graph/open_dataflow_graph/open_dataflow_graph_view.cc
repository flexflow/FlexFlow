#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"

namespace FlexFlow {

std::unordered_set<DataflowGraphInput>
    OpenDataflowGraphView::get_inputs() const {
  return this->get_interface().get_inputs();
}

std::unordered_set<OpenDataflowEdge>
    OpenDataflowGraphView::query_edges(OpenDataflowEdgeQuery const &q) const {
  return this->get_interface().query_edges(q);
}

IOpenDataflowGraphView const &OpenDataflowGraphView::get_interface() const {
  return *std::dynamic_pointer_cast<IOpenDataflowGraphView const>(
      GraphView::ptr.get());
}

} // namespace FlexFlow

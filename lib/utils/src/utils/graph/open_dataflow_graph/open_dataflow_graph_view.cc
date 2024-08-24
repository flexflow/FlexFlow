#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/algorithms/as_dot.h"

namespace FlexFlow {

std::unordered_set<DataflowGraphInput>
    OpenDataflowGraphView::get_inputs() const {
  return this->get_interface().get_inputs();
}

std::unordered_set<OpenDataflowEdge>
    OpenDataflowGraphView::query_edges(OpenDataflowEdgeQuery const &q) const {
  return this->get_interface().query_edges(q);
}

void OpenDataflowGraphView::debug_print_dot() const {
  std::cout << as_dot(*this) << std::endl;
}

IOpenDataflowGraphView const &OpenDataflowGraphView::get_interface() const {
  return *std::dynamic_pointer_cast<IOpenDataflowGraphView const>(
      GraphView::ptr.get());
}

} // namespace FlexFlow

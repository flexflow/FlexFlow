#include "utils/graph/open_dataflow_graph/open_dataflow_graph.h"

namespace FlexFlow {

NodeAddedResult
    OpenDataflowGraph::add_node(std::vector<OpenDataflowValue> const &inputs,
                                int num_outputs) {
  return this->get_interface().add_node(inputs, num_outputs);
}

DataflowGraphInput OpenDataflowGraph::add_input() {
  return this->get_interface().add_input();
}

IOpenDataflowGraph &OpenDataflowGraph::get_interface() {
  return *std::dynamic_pointer_cast<IOpenDataflowGraph>(
      GraphView::ptr.get_mutable());
}

IOpenDataflowGraph const &OpenDataflowGraph::get_interface() const {
  return *std::dynamic_pointer_cast<IOpenDataflowGraph const>(
      GraphView::ptr.get());
}

} // namespace FlexFlow

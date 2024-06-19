#include "utils/graph/dataflow_graph/dataflow_graph.h"

namespace FlexFlow {

NodeAddedResult DataflowGraph::add_node(std::vector<DataflowOutput> const &inputs,
                                        int num_outputs) {
  return this->get_interface().add_node(inputs, num_outputs);
}

} // namespace FlexFlow

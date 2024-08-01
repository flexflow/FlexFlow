#include "utils/graph/dataflow_graph/dataflow_graph.h"

namespace FlexFlow {

NodeAddedResult
    DataflowGraph::add_node(std::vector<DataflowOutput> const &inputs,
                            int num_outputs) {
  return this->get_interface().add_node(inputs, num_outputs);
}

void DataflowGraph::add_node_unsafe(
    Node const &node,
    std::vector<DataflowOutput> const &inputs,
    std::vector<DataflowOutput> const &outputs) {
  return this->get_interface().add_node_unsafe(node, inputs, outputs);
}

std::unordered_set<Node> DataflowGraph::query_nodes(NodeQuery const &q) const {
  return this->get_interface().query_nodes(q);
}

std::unordered_set<DataflowEdge>
    DataflowGraph::query_edges(DataflowEdgeQuery const &q) const {
  return this->get_interface().query_edges(q);
}

std::unordered_set<DataflowOutput>
    DataflowGraph::query_outputs(DataflowOutputQuery const &q) const {
  return this->get_interface().query_outputs(q);
}

IDataflowGraph &DataflowGraph::get_interface() {
  return *std::dynamic_pointer_cast<IDataflowGraph>(
      GraphView::ptr.get_mutable());
}

IDataflowGraph const &DataflowGraph::get_interface() const {
  return *std::dynamic_pointer_cast<IDataflowGraph const>(GraphView::ptr.get());
}

} // namespace FlexFlow

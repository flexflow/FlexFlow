#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_OPERATOR_GRAPH_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_OPERATOR_GRAPH_DATAFLOW_GRAPH_H

#include "utils/containers/enumerate_vector.h"
#include "utils/graph.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel>
struct DataflowGraph {
public:  
  DataflowGraph() 
    : g(OutputLabelledMultiDiGraph<NodeLabel, OutputLabel>::template create<
          UnorderedOutputLabelledMultiDiGraph<NodeLabel, OutputLabel>>()) { }

  std::vector<MultiDiOutput> add_operator(NodeLabel const &func, std::vector<MultiDiOutput> const &inputs, std::vector<OutputLabel> const &outputs) {
    Node n = this->g.add_node(func);
    for (auto const &[idx, input] : enumerate_vector(inputs)) {
      this->g.add_edge(MultiDiEdge{input.src, input.src_idx, n, this->make_port_for_idx(idx)});
    }

    std::vector<MultiDiOutput> result;
    for (auto const &[idx, label] : enumerate_vector(outputs)) {
      MultiDiOutput output = MultiDiOutput{n, this->make_port_for_idx(idx)};
      this->g.add_output(output, label);
      result.push_back(output);
    }

    return result;
  }

  NodePort make_port_for_idx(int idx) {
    if (!this->port_mapping.contains_l(idx)) {
      this->port_mapping.equate(idx, this->g.add_node_port());
    } 
    return this->port_mapping.at_l(idx);
  }

  NodePort port_for_idx(int idx) const {
    return this->port_mapping.at_l(idx);
  }

  int idx_for_port(NodePort const &p) const {
    return this->port_mapping.at_r(p); 
  }

  OutputLabelledMultiDiGraphView<NodeLabel, OutputLabel> const &get_raw_graph() const {
    return this->g;
  }

  NodeLabel const &at(Node const &n) const {
    return this->g.at(n);
  }

  OutputLabel const &at(MultiDiOutput const &o) const {
    return this->g.at(o);
  }
private:
  OutputLabelledMultiDiGraph<NodeLabel, OutputLabel> g;
  bidict<int, NodePort> port_mapping;
};

template <typename NodeLabel, typename OutputLabel>
std::unordered_set<Node> get_nodes(DataflowGraph<NodeLabel, OutputLabel> const &g) {
  return get_nodes(g.get_raw_graph());
}

} // namespace FlexFlow

#endif

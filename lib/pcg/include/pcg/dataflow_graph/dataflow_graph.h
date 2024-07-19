#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_OPERATOR_GRAPH_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_OPERATOR_GRAPH_DATAFLOW_GRAPH_H

// #include "pcg/dataflow_graph/operator_added_result.dtg.h"
// #include "utils/containers/enumerate_vector.h"
// #include "utils/graph.h"
//
// namespace FlexFlow {
//
// template <typename NodeLabel, typename OutputLabel>
// struct DataflowGraph {
// public:
//   DataflowGraph()
//       : g(OutputLabelledMultiDiGraph<NodeLabel, OutputLabel>::template
//       create<
//             UnorderedOutputLabelledMultiDiGraph<NodeLabel, OutputLabel>>())
//             {}
//
//   OperatorAddedResult
//       add_operator(NodeLabel const &func,
//                    std::vector<MultiDiOutput> const &inputs,
//                    std::vector<OutputLabel> const &output_labels) {
//     Node node = this->g.add_node(func);
//     for (auto const &[idx, input] : enumerate_vector(inputs)) {
//       this->g.add_edge(MultiDiEdge{
//           node, this->make_port_for_idx(idx), input.src, input.src_idx});
//     }
//
//     std::vector<MultiDiOutput> outputs;
//     for (auto const &[idx, label] : enumerate_vector(output_labels)) {
//       MultiDiOutput output = MultiDiOutput{node,
//       this->make_port_for_idx(idx)}; this->g.add_output(output, label);
//       outputs.push_back(output);
//     }
//     this->output_map[node] = outputs;
//
//     return OperatorAddedResult{
//         node,
//         outputs,
//     };
//   }
//
//   NodePort make_port_for_idx(int idx) {
//     if (!this->port_mapping.contains_l(idx)) {
//       this->port_mapping.equate(idx, this->g.add_node_port());
//     }
//     return this->port_mapping.at_l(idx);
//   }
//
//   NodePort port_for_idx(int idx) const {
//     return this->port_mapping.at_l(idx);
//   }
//
//   int idx_for_port(NodePort const &p) const {
//     return this->port_mapping.at_r(p);
//   }
//
//   OutputLabelledMultiDiGraphView<NodeLabel, OutputLabel> const &
//       get_raw_graph() const {
//     return this->g;
//   }
//
//   NodeLabel const &at(Node const &n) const {
//     return this->g.at(n);
//   }
//
//   OutputLabel const &at(MultiDiOutput const &o) const {
//     return this->g.at(o);
//   }
//
//   std::unordered_map<Node, std::vector<MultiDiOutput>> const &
//       get_output_map() const {
//     return this->output_map;
//   }
//
// private:
//   OutputLabelledMultiDiGraph<NodeLabel, OutputLabel> g;
//   bidict<int, NodePort> port_mapping;
//   std::unordered_map<Node, std::vector<MultiDiOutput>>
//       output_map; // NOTE(@lockshaw): temporary workaround until not tracking
//                   // outputs independent of edges in multidigraph is resolved
// };
//
// template <typename NodeLabel, typename OutputLabel>
// std::unordered_set<Node>
//     get_nodes(DataflowGraph<NodeLabel, OutputLabel> const &g) {
//   return get_nodes(g.get_raw_graph());
// }
//
// template <typename T>
// std::vector<T>
//     vector_from_indexed_set(std::vector<std::pair<int, T>> const &s) {
//   std::vector<std::optional<T>> result{s.size(), std::nullopt};
//   for (auto const &[idx, value] : s) {
//     assert(idx < s.size() && idx >= 0);
//     assert(!result.at(idx).has_value());
//     result.at(idx) = value;
//   }
//   return transform(result, [](std::optional<T> const &v) {
//     assert(v.has_value());
//     return v.value();
//   });
// }
//
// } // namespace FlexFlow

#endif

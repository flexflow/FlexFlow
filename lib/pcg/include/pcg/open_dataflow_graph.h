// #ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_OPEN_DATAFLOW_GRAPH_H
// #define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_OPEN_DATAFLOW_GRAPH_H
// 
// #include "utils/containers/enumerate_vector.h"
// #include "utils/graph.h"
// #include "pcg/dataflow_input.dtg.h"
// 
// namespace FlexFlow {
// 
// template <typename NodeLabel, typename OutputLabel>
// struct OpenDataflowGraph {
// public:
//   OpenDataflowGraph()
//     : g(OutputLabelledOpenMultiDiGraph<NodeLabel, OutputLabel>::template create<
//           UnorderedOutputLabelledOpenMultiDiGraph<NodeLabel, OutputLabel>>()) { }
// 
//   DataflowInput add_external_input(OutputLabel const &label) {
//     /* size_t src_node_idx = edge_uid_ctr; */
//     /* edge_uid_ctr++; */
//     /* size_t src_port_idx = 0; */
//     /* edge_uid_t edge_uid = { src_node_idx, src_port_idx }; */
//     /* return MultiDiOutput{edge_uid}; */
//   }
// 
//   std::vector<MultiDiOutput> add_operator(NodeLabel const &func, std::vector<DataflowInput> const &inputs, std::vector<OutputLabel> const &outputs) {
//     Node n = this->g.add_node(func);
//     for (auto const &[idx, input] : enumerate_vector(inputs)) {
//       this->g.add_edge(MultiDiEdge{input.src, input.src_idx, n, this->make_port_for_idx(idx)});
//     }
// 
//     std::vector<MultiDiOutput> result;
//     for (auto const &[idx, label] : enumerate_vector(outputs)) {
//       MultiDiOutput output = MultiDiOutput{n, this->make_port_for_idx(idx)};
//       this->g.add_output(output, label);
//       result.push_back(output);
//     }
// 
//     return result;
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
//   OutputLabelledMultiDiGraphView<NodeLabel, OutputLabel> const &get_raw_graph() const {
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
// private:
//   OutputLabelledOpenMultiDiGraph<NodeLabel, OutputLabel> g;
//   bidict<int, NodePort> port_mapping;
//   size_t edge_uid_ctr = 0;
// };
// 
// } // namespace FlexFlow
// 
// #endif

#include "pcg/computation_graph.h"

namespace FlexFlow {

std::vector<operator_guid_t>
    traverse_comp_graph_forward(ComputationGraph const &comp_graph) {
  std::vector<Node> layers = get_topological_ordering(comp_graph.value());
  return transform(layers, [&](Node const &e) -> operator_guid_t {
    return operator_guid_t{e};
  });
}

std::vector<operator_guid_t>
    traverse_comp_graph_backward(ComputationGraph const &comp_graph) {
  std::vector<Node> layers =
      reversed<std::vector<Node>>(get_topological_ordering(comp_graph.value()));
  return transform(layers, [&](Node const &e) -> operator_guid_t {
    return operator_guid_t{e};
  });
}

std::vector<tensor_guid_t>
    sort_edge_set(std::unordered_set<MultiDiEdge> edges) {
  return transform(
      sorted_by(edges, compare_by<MultiDiEdge>([](MultiDiEdge const &e) {
                  return e.src_idx;
                })),
      [&](MultiDiEdge const &e) -> tensor_guid_t { return tensor_guid_t{e}; });
}

std::vector<tensor_guid_t>
   get_outgoing_tensors(ComputationGraph const &comp_graph,
                         operator_guid_t n) {
  return sort_edge_set(get_outgoing_edges(comp_graph.value(), n.value()));
}

std::vector<tensor_guid_t>
    get_incoming_tensors(ComputationGraph const &comp_graph,
                         operator_guid_t n) {
  return sort_edge_set(get_incoming_edges(comp_graph.value(), n.value()));
}

operator_guid_t create_node(ComputationGraph &comp_graph, Layer const &layer) {
  Node added_node = comp_graph.value().add_node(layer);
  return operator_guid_t{added_node};
}

tensor_guid_t create_outgoing_edge(ComputationGraph &comp_graph,
                                   operator_guid_t node,
                                   int idx,
                                   Tensor tensor) {
  MultiDiOutput edge = {node.value(), NodePort{idx}};
  comp_graph.value().add_output(edge, tensor);
  return tensor_guid_t{edge};
}

void connect_incoming_edges(ComputationGraph &comp_graph,
                            std::vector<tensor_guid_t> const &incoming_edges,
                            operator_guid_t node) {
  size_t incoming_edge_dst_port = 0;
  for (tensor_guid_t input : incoming_edges) {
    MultiDiOutput input_view = input.value();
    MultiDiEdge edge = {node.value(),
                        NodePort{incoming_edge_dst_port++},
                        input_view.src,
                        input_view.src_idx};
    comp_graph.value().add_edge(edge);
  }
}

CompGraphOperatorAttrs get_layer_attrs(ComputationGraph const &comp_graph,
                                       operator_guid_t const &n) {
  return comp_graph.at(n).attrs;
}

} // namespace FlexFlow

#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_H

#include "layer.h"
#include "operator_guid_t.h"
#include "tensor.h"
#include "tensor_guid_t.h"
#include "utils/containers.h"
#include "utils/graph.h"
#include "utils/strong_typedef.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

struct ComputationGraph
    : public strong_typedef<ComputationGraph,
                            OutputLabelledMultiDiGraph<Layer, Tensor>> {
  using strong_typedef::strong_typedef;

  std::vector<operator_guid_t> traverse() {
    std::vector<Node> layers = get_topological_ordering(this->value());
    return transform(layers, [&](Node const &e) -> operator_guid_t {
      return operator_guid_t{e};
    });
  }

  std::vector<operator_guid_t> traverse_reverse_order() {
    std::vector<Node> layers =
        reversed<std::vector<Node>>(get_topological_ordering(this->value()));
    return transform(layers, [&](Node const &e) -> operator_guid_t {
      return operator_guid_t{e};
    });
  }

  bool out_edge_comparator(MultiDiOutput x, MultiDiOutput y) {
    return x.src_idx < y.src_idx;
  }

  std::vector<tensor_guid_t>
      sort_edge_set(std::unordered_set<MultiDiEdge> edges) {
    std::unordered_set<MultiDiOutput> outputs =
        transform(edges, [&](MultiDiEdge const &e) -> MultiDiOutput {
          return MultiDiOutput(e);
        });
    std::vector<MultiDiOutput> sorted_outputs(outputs.begin(), outputs.end());
    sort(sorted_outputs.begin(), sorted_outputs.end(), out_edge_comparator);
    return transform(sorted_outputs,
                     [&](MultiDiOutput const &e) -> tensor_guid_t {
                       return tensor_guid_t{e};
                     });
  }

  std::vector<tensor_guid_t> get_outgoing_tensors(operator_guid_t n) {
    return sort_edge_set(get_outgoing_edges(this->value(), n.value()));
  }

  std::vector<tensor_guid_t> get_incoming_tensors(operator_guid_t n) {
    return sort_edge_set(get_incoming_edges(this->value(), n.value()));
  }

  operator_guid_t add_node(Layer const &layer) {
    Node added_node = this->value().add_node(layer);
    return operator_guid_t{added_node};
  }

  void add_output(tensor_guid_t const &output, Tensor const &tensor) {
    this->value().add_output(output.value(), tensor);
  }

  tensor_guid_t create_outgoing_edge(operator_guid_t node, int idx) {
    MultiDiOutput edge = {node.value(), NodePort{idx}};
    return tensor_guid_t{edge};
  }

  tensor_guid_t create_outgoing_edge_with_label(operator_guid_t node,
                                                int idx,
                                                Tensor tensor) {
    tensor_guid_t tensor_guid = create_outgoing_edge(node, idx);
    add_output(tensor_guid, tensor);
    return tensor_guid;
  }

  void add_incoming_edges(std::vector<tensor_guid_t> const &incoming_edges,
                          operator_guid_t node) {
    size_t incoming_edge_dst_port = 0;
    for (tensor_guid_t input : incoming_edges) {
      MultiDiOutput input_view = input.value();
      MultiDiEdge edge = {node.value(),
                          NodePort{incoming_edge_dst_port++},
                          input_view.src,
                          input_view.src_idx};
      this->value().add_edge(edge);
    }
  }

  Layer &at(operator_guid_t const &n) {
    return this->value().at(n.value());
  }

  Layer const &at(operator_guid_t const &n) const {
    return this->value().at(n.value());
  }

  Tensor &at(tensor_guid_t const &e) {
    return this->value().at(e.value());
  }

  Tensor const &at(tensor_guid_t const &e) const {
    return this->value().at(e.value());
  }

  CompGraphOperatorAttrs get_layer_attrs(operator_guid_t const &n) const {
    return this->at(n).attrs;
  }
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_HASH(ComputationGraph);

} // namespace FlexFlow

#endif

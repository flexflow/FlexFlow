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
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_HASH(ComputationGraph);

std::vector<operator_guid_t>
    topological_ordering(ComputationGraph const &comp_graph);
std::vector<operator_guid_t>
    reverse_topological_ordering(ComputationGraph const &comp_graph);
std::vector<tensor_guid_t>
    get_outgoing_tensors(ComputationGraph const &comp_graph, operator_guid_t n);
std::vector<tensor_guid_t>
    get_incoming_tensors(ComputationGraph const &comp_graph, operator_guid_t n);
operator_guid_t create_node(ComputationGraph &comp_graph, Layer const &layer);
tensor_guid_t create_outgoing_edge(ComputationGraph &comp_graph,
                                   operator_guid_t node,
                                   int idx,
                                   Tensor tensor);

void connect_incoming_edges(ComputationGraph &comp_graph,
                            std::vector<tensor_guid_t> const &incoming_edges,
                            operator_guid_t node);
CompGraphOperatorAttrs get_layer_attrs(ComputationGraph const &comp_graph,
                                       operator_guid_t const &n);

} // namespace FlexFlow

#endif

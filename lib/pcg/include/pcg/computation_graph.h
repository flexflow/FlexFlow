#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_H

#include "pcg/computation_graph.dtg.h"
#include "pcg/computation_graph/layer_added_result.dtg.h"
#include "pcg/layer_guid_t.dtg.h"
#include "pcg/tensor_attrs.dtg.h"
#include "pcg/tensor_guid_t.dtg.h"
#include "pcg/computation_graph/computation_graph_edge.dtg.h"

namespace FlexFlow {

ComputationGraph make_empty_computation_graph();

std::unordered_set<layer_guid_t> get_layers(ComputationGraph const &);

LayerAddedResult add_layer(ComputationGraph &computation_graph,
                           LayerAttrs const &attrs,
                           std::vector<tensor_guid_t> const &inputs,
                           std::vector<TensorAttrs> const &outputs);
TensorAttrs get_tensor_attrs(ComputationGraph const &, tensor_guid_t const &);
bool are_tensor_guid_shapes_equivalent(ComputationGraph const &cg,
                                       tensor_guid_t const &t1,
                                       tensor_guid_t const &t2);

std::vector<layer_guid_t> topological_ordering(ComputationGraph const &cg);

std::vector<tensor_guid_t> get_outgoing_tensors(ComputationGraph const &cg,
                                                layer_guid_t n);

std::vector<tensor_guid_t> get_incoming_tensors(ComputationGraph const &cg,
                                                layer_guid_t n);

std::unordered_set<ComputationGraphEdge> get_subgraph_incoming_edges(
    ComputationGraph const &,
    std::unordered_set<layer_guid_t> const &);
std::unordered_set<ComputationGraphEdge> get_subgraph_outgoing_edges(
    ComputationGraph const &,
    std::unordered_set<layer_guid_t> const &);

LayerAttrs get_layer_attrs(ComputationGraph const &cg, layer_guid_t const &n);

layer_guid_t get_layer_by_name(ComputationGraph const &cg,
                               std::string const &name);

} // namespace FlexFlow

#endif

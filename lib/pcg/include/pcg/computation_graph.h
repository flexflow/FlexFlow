#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_H

#include "pcg/computation_graph.dtg.h"
#include "pcg/computation_graph/layer_added_result.dtg.h"
#include "pcg/layer_guid_t.dtg.h"
#include "pcg/tensor_attrs.dtg.h"
#include "pcg/tensor_guid_t.dtg.h"

namespace FlexFlow {

ComputationGraph make_empty_computation_graph();

std::unordered_set<layer_guid_t> get_layers(ComputationGraph const &);

LayerAddedResult add_layer(ComputationGraph &computation_graph,
                           LayerAttrs const &attrs,
                           std::vector<tensor_guid_t> const &inputs,
                           std::vector<TensorAttrs> const &outputs);
TensorAttrs get_tensor_attrs(ComputationGraph const &, tensor_guid_t const &);

std::vector<layer_guid_t> topological_ordering(ComputationGraph const &cg);

std::vector<tensor_guid_t> get_outgoing_tensors(ComputationGraph const &cg,
                                                layer_guid_t n);

std::vector<tensor_guid_t> get_incoming_tensors(ComputationGraph const &cg,
                                                layer_guid_t n);

LayerAttrs get_layer_attrs(ComputationGraph const &cg, layer_guid_t const &n);

layer_guid_t get_layer_by_name(ComputationGraph const &cg,
                               std::string const &name);

std::vector<tensor_guid_t>
    get_new_tensor_guids_for_layer_without_graph_insertion(
        ComputationGraph const &, layer_guid_t const &n, int num_tensors);

} // namespace FlexFlow

#endif

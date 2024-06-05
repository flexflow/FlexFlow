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

std::vector<layer_guid_t>
    reverse_topological_ordering(ComputationGraph const &cg);

std::vector<tensor_guid_t> get_outgoing_tensors(ComputationGraph const &cg,
                                                layer_guid_t n);

std::vector<tensor_guid_t> get_incoming_tensors(ComputationGraph const &cg,
                                                layer_guid_t n);

ComputationGraphOpAttrs get_layer_attrs(ComputationGraph const &cg,
                                        layer_guid_t const &n);

} // namespace FlexFlow

#endif

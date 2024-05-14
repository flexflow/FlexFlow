#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_H

#include "pcg/computation_graph.dtg.h"
#include "pcg/tensor_guid_t.dtg.h"
#include "pcg/tensor_attrs.dtg.h"
#include "pcg/layer_guid_t.dtg.h"
#include "pcg/computation_graph/layer_added_result.dtg.h"

namespace FlexFlow {

ComputationGraph make_empty_computation_graph();

std::unordered_set<layer_guid_t> get_layers(ComputationGraph const &);

LayerAddedResult add_layer(ComputationGraph &computation_graph, LayerAttrs const &attrs, std::vector<tensor_guid_t> const &inputs, std::vector<TensorAttrs> const &outputs);
TensorAttrs get_tensor_attrs(ComputationGraph const &, tensor_guid_t const &);

} // namespace FlexFlow

#endif

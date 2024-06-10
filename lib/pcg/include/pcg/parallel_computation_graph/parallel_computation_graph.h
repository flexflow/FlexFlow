#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_H

#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"

namespace FlexFlow {

ParallelComputationGraph empty_parallel_computation_graph();

std::unordered_set<parallel_layer_guid_t>
    get_parallel_layers(ParallelComputationGraph const &);

std::vector<parallel_tensor_guid_t>
    get_layer_inputs(ParallelComputationGraph const &,
                     parallel_layer_guid_t const &);
std::vector<parallel_tensor_guid_t>
    get_layer_outputs(ParallelComputationGraph const &,
                      parallel_layer_guid_t const &);

parallel_layer_guid_t get_source_layer(ParallelComputationGraph const &,
                                       parallel_tensor_guid_t const &);

ParallelLayerAttrs get_parallel_layer_attrs(ParallelComputationGraph const &,
                                            parallel_layer_guid_t const &);
ParallelTensorAttrs get_parallel_tensor_attrs(ParallelComputationGraph const &,
                                              parallel_tensor_guid_t const &);

} // namespace FlexFlow

#endif

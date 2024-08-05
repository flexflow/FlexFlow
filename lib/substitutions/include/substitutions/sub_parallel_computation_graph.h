#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_SUB_PARALLEL_COMPUTATION_GRAPH_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_SUB_PARALLEL_COMPUTATION_GRAPH_H

#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"
#include "substitutions/open_parallel_tensor_guid_t.dtg.h"
#include "substitutions/sub_parallel_computation_graph.dtg.h"
#include "substitutions/sub_parallel_computation_graph_data.dtg.h"

namespace FlexFlow {

std::unordered_set<parallel_layer_guid_t>
    get_parallel_layers(SubParallelComputationGraph const &sub_pcg);
ParallelLayerAttrs get_parallel_layer_attrs(SubParallelComputationGraph const &,
                                            parallel_layer_guid_t const &);
PCGOperatorAttrs get_operator_attrs(SubParallelComputationGraph const &,
                                    parallel_layer_guid_t const &);
ParallelTensorAttrs
    get_parallel_tensor_attrs(SubParallelComputationGraph const &,
                              open_parallel_tensor_guid_t const &);
SubParallelComputationGraph
    sub_pcg_from_full_pcg(ParallelComputationGraph const &);
ParallelComputationGraph
    pcg_from_sub_pcg_by_dropping_inputs(SubParallelComputationGraph const &);

parallel_layer_guid_t
    get_parallel_layer_by_name(SubParallelComputationGraph const &pcg,
                               std::string const &name);

std::vector<open_parallel_tensor_guid_t>
    get_layer_inputs(SubParallelComputationGraph const &,
                     parallel_layer_guid_t const &);
std::vector<parallel_tensor_guid_t>
    get_layer_outputs(SubParallelComputationGraph const &,
                      parallel_layer_guid_t const &);

SubParallelComputationGraphData get_graph_data(SubParallelComputationGraph const &);

} // namespace FlexFlow

#endif

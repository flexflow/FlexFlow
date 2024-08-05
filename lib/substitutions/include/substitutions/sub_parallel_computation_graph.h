#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_SUB_PARALLEL_COMPUTATION_GRAPH_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_SUB_PARALLEL_COMPUTATION_GRAPH_H

#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "substitutions/sub_parallel_computation_graph.dtg.h"

namespace FlexFlow {

std::unordered_set<parallel_layer_guid_t>
    get_parallel_layers(SubParallelComputationGraph const &sub_pcg);
ParallelLayerAttrs get_parallel_layer_attrs(SubParallelComputationGraph const &,
                                            Node const &);
PCGOperatorAttrs get_operator_attrs(SubParallelComputationGraph const &,
                                    Node const &);
ParallelTensorAttrs
    get_parallel_tensor_attrs(SubParallelComputationGraph const &,
                              OpenDataflowValue const &);
SubParallelComputationGraph
    sub_pcg_from_full_pcg(ParallelComputationGraph const &);
ParallelComputationGraph
    pcg_from_sub_pcg_by_dropping_inputs(SubParallelComputationGraph const &);

SubParallelComputationGraph
    sub_pcg_from_partial_pcg(ParallelComputationGraph const &, std::unordered_set<Node> const &);

parallel_layer_guid_t
    get_parallel_layer_by_name(SubParallelComputationGraph const &pcg,
                               std::string const &name);

} // namespace FlexFlow

#endif

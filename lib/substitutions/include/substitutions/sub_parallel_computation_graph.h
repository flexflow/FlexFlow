#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_SUB_PARALLEL_COMPUTATION_GRAPH_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_SUB_PARALLEL_COMPUTATION_GRAPH_H

#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_use_t.dtg.h"
#include "substitutions/open_parallel_tensor_guid_t.dtg.h"
#include "substitutions/sub_parallel_computation_graph.dtg.h"
#include "substitutions/sub_parallel_computation_graph_data.dtg.h"
#include "substitutions/sub_parallel_computation_graph_edge.dtg.h"

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

std::unordered_set<SubParallelComputationGraphEdge>
    get_subgraph_incoming_edges(SubParallelComputationGraph const &,
                       std::unordered_set<parallel_layer_guid_t> const &);
std::unordered_set<ParallelComputationGraphEdge>
    get_subgraph_outgoing_edges(SubParallelComputationGraph const &,
                       std::unordered_set<parallel_layer_guid_t> const &);

std::unordered_set<SubParallelComputationGraphEdge> get_subgraph_incoming_edges(
    SubParallelComputationGraph const &,
    std::unordered_set<parallel_layer_guid_t> const &);

std::unordered_set<parallel_tensor_use_t>
    get_parallel_tensor_uses(SubParallelComputationGraph const &,
                             open_parallel_tensor_guid_t const &);

SubParallelComputationGraphData
    get_sub_pcg_data(SubParallelComputationGraph const &);
SubParallelComputationGraph
    sub_pcg_from_graph_data(SubParallelComputationGraphData const &);
bool are_isomorphic(SubParallelComputationGraph const &,
                    SubParallelComputationGraph const &);

SubParallelComputationGraph
    without_layer_names(SubParallelComputationGraph const &);

std::string as_dot(SubParallelComputationGraph const &);
void debug_print_dot(SubParallelComputationGraph const &);

} // namespace FlexFlow

#endif

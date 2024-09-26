#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_CONTEXT_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_CONTEXT_H

#include "compiler/machine_mapping/machine_mapping_context.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"

namespace FlexFlow {

std::unordered_set<MachineView> get_allowed_machine_views_for_tensor(MachineMappingContext const &,
                                                                     parallel_tensor_guid_t const &);
std::unordered_set<MachineView> get_allowed_machine_views_for_layer(MachineMappingContext const &,
                                                                    parallel_layer_guid_t const &);

MachineMappingContext make_machine_mapping_context(ParallelComputationGraph const &pcg,
                                                   CostEstimator const &cost_estimator,
                                                   std::function<std::unordered_set<MachineView>(
                                                     ParallelLayerAttrs const &, MachineSpecification const &)> const &allowed_machine_views);

} // namespace FlexFlow

#endif

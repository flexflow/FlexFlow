#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_GET_OPTIMAL_MACHINE_MAPPING_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_GET_OPTIMAL_MACHINE_MAPPING_H

#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/machine_mapping/machine_mapping_cache.h"
#include "compiler/machine_mapping/machine_mapping_context.dtg.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"

namespace FlexFlow {

MachineMappingResult get_optimal_machine_mapping(
    ParallelComputationGraph const &pcg,
    std::function<std::unordered_set<MachineView>(
        ParallelLayerAttrs const &, MachineSpecification const &)> const
        &allowed_machine_views,
    CostEstimator const &cost_estimator,
    MachineSpecification const &resources,
    MachineMappingCache &cached_subgraph_results);

MachineMappingResult
    get_optimal_machine_mapping_internal(MachineMappingContext &context,
                                         MachineSpecification const &resources);

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingContext &context,
    SerialParallelDecomposition const &decompn,
    MachineSpecification const &resource,
    std::unordered_map<OpenDataflowValue, MachineView> const
        &fixed_machine_views);

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingContext &context,
    SerialSplit const &serial,
    MachineSpecification const &resource,
    std::unordered_map<OpenDataflowValue, MachineView> const
        &fixed_machine_views);

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingContext &context,
    ParallelSplit const &parallel,
    MachineSpecification const &resource,
    std::unordered_map<OpenDataflowValue, MachineView> const
        &fixed_machine_views);

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingContext &context,
    Node const &node,
    MachineSpecification const &resource,
    std::unordered_map<OpenDataflowValue, MachineView> const
        &fixed_machine_views);

} // namespace FlexFlow

#endif

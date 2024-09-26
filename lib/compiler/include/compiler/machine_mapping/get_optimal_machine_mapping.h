#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_GET_OPTIMAL_MACHINE_MAPPING_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_GET_OPTIMAL_MACHINE_MAPPING_H

#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/machine_mapping/machine_mapping_cache.h"
#include "compiler/machine_mapping/machine_mapping_context.dtg.h"
#include "compiler/machine_mapping/partial_machine_mapping.dtg.h"
#include "compiler/series_parallel/pcg_binary_parallel_split.dtg.h"
#include "compiler/series_parallel/pcg_binary_series_split.dtg.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.h"

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
    get_optimal_machine_mapping_internal(MachineMappingCache &result_cache,
                                         MachineMappingContext const &context,
                                         MachineSpecification const &resources);

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingCache &result_cache,
    MachineMappingContext const &context,
    PCGBinarySPDecomposition const &sp_decomposition,
    MachineSpecification const &resources,
    PartialMachineMapping const &);

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingCache &result_cache,
    MachineMappingContext const &context,
    PCGBinarySeriesSplit const &series,
    MachineSpecification const &resources,
    PartialMachineMapping const &);

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingCache &result_cache,
    MachineMappingContext const &context,
    PCGBinaryParallelSplit const &parallel,
    MachineSpecification const &resources,
    PartialMachineMapping const &);

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingCache &result_cache,
    MachineMappingContext const &,
    parallel_layer_guid_t const &,
    MachineSpecification const &,
    PartialMachineMapping const &);

} // namespace FlexFlow

#endif

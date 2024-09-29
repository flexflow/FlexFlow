#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_GET_OPTIMAL_MACHINE_MAPPING_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_GET_OPTIMAL_MACHINE_MAPPING_H

#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/machine_mapping/machine_mapping_cache.h"
#include "compiler/machine_mapping/machine_mapping_constraints.dtg.h"
#include "compiler/machine_mapping/machine_mapping_context.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.dtg.h"
#include "compiler/machine_mapping/machine_mapping_result_tree/machine_mapping_result_tree.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/mm_problem_tree_parallel_split.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/mm_problem_tree_series_split.dtg.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.h"

namespace FlexFlow {

MachineMappingResultTree get_optimal_machine_mapping(
    ParallelComputationGraph const &pcg,
    std::function<std::unordered_set<MachineView>(
        ParallelLayerAttrs const &, MachineSpecification const &)> const
        &allowed_machine_views,
    CostEstimator const &cost_estimator,
    MachineSpecification const &resources,
    MachineMappingCache &cached_subgraph_results);

MachineMappingResultTree
    get_optimal_machine_mapping_internal(MachineMappingCache &result_cache,
                                         MachineMappingContext const &context,
                                         MachineSpecification const &resources);

std::optional<MachineMappingResultTree> get_optimal_machine_mapping_internal(
    MachineMappingCache &result_cache,
    MachineMappingContext const &context,
    MachineMappingProblemTree const &,
    MachineSpecification const &resources,
    MachineMappingConstraints const &);

std::optional<MachineMappingResultTree> get_optimal_machine_mapping_internal(
    MachineMappingCache &result_cache,
    MachineMappingContext const &context,
    MMProblemTreeSeriesSplit const &,
    MachineSpecification const &resources,
    MachineMappingConstraints const &);

std::optional<MachineMappingResultTree> get_optimal_machine_mapping_internal(
    MachineMappingCache &result_cache,
    MachineMappingContext const &context,
    MMProblemTreeParallelSplit const &,
    MachineSpecification const &resources,
    MachineMappingConstraints const &);

std::optional<MachineMappingResultTree> get_optimal_machine_mapping_internal(
    MachineMappingCache &result_cache,
    MachineMappingContext const &,
    parallel_layer_guid_t const &,
    MachineSpecification const &,
    MachineMappingConstraints const &);

} // namespace FlexFlow

#endif

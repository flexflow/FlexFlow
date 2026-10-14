#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_UNITY_ALGORITHM_UNITY_ALGORITHM_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_UNITY_ALGORITHM_UNITY_ALGORITHM_H

#include "compiler/cost_estimator/cost_estimator.h"
#include "compiler/cost_estimator/runtime_only_cost_estimator.h"
#include "compiler/search_result.dtg.h"
#include "compiler/unity_algorithm/unity_search_config.dtg.h"
#include "pcg/machine_specification.dtg.h"
#include "substitutions/substitution.h"

namespace FlexFlow {

std::vector<ParallelComputationGraph>
    all_pcgs_obtained_by_applying_a_substitution(
        ParallelComputationGraph const &pcg,
        std::vector<Substitution> const &substitutions);

SearchResult optimize_with_unity_algorithm(
    ParallelComputationGraph &pcg,
    RuntimeOnlyCostEstimator const &cost_estimator,
    MachineComputeSpecification const &resources,
    UnitySearchConfig const &search_config,
    std::vector<Substitution> const &substitutions);

} // namespace FlexFlow

#endif

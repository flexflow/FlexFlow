#ifndef _FLEXFLOW_COMPILER_UNITY_ALGORITHM_H
#define _FLEXFLOW_COMPILER_UNITY_ALGORITHM_H

#include "compiler/graph_optimize_result.dtg.h"
#include "compiler/machine_mapping.h"
#include "cost_estimator.h"
#include "pcg/computation_graph.h"
#include "pcg/machine_specification.dtg.h"
#include "substitutions/sub_parallel_computation_graph.h"

namespace FlexFlow {

struct OptimizerConfig {
  float alpha;
  int budget;
  float threshold;
  int max_num_ops;
};

GraphOptimizeResult graph_optimize(
    ParallelComputationGraph &pcg,
    CostEstimator const &cost_estimator,
    MachineSpecification const &resources,
    std::function<std::unordered_set<MachineView>(
        ParallelLayerAttrs const &, MachineSpecification const &)> const
        &allowed_machine_views,
    OptimizerConfig const &opt_config);

} // namespace FlexFlow

#endif

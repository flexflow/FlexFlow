#ifndef _FLEXFLOW_COMPILER_UNITY_ALGORITHM_H
#define _FLEXFLOW_COMPILER_UNITY_ALGORITHM_H

#include "cost_estimate.h"
#include "machine_mapping.h"
#include "pcg/computation_graph.h"
#include "sub_parallel_computation_graph.h"

namespace FlexFlow {

struct Substitution {};

struct Strategy {
  ParallelComputationGraph pcg;
  MachineMapping machine_mapping;
  req<float> runtime;
};
FF_VISITABLE_STRUCT(Strategy, pcg, machine_mapping, runtime);

struct StrategyRuntimeCmp {
  bool operator()(Strategy const &, Strategy const &);
};

struct OptimizerConfig {
  float alpha;
  int budget;
  float threshold;
  int max_num_ops;
};

Strategy
    graph_optimize(ComputationGraph &cg,
                   ICostEstimator const &cost_estimator,
                   MachineSpecification const &resources,
                   std::function<std::unordered_set<MachineView>(
                       Operator const &, MachineSpecification const &)> const
                       &allowed_machine_views,
                   OptimizerConfig const &opt_config);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_COMPILER_UNITY_ALGORITHM_H
#define _FLEXFLOW_COMPILER_UNITY_ALGORITHM_H

#include "cost_estimate.h"
#include "machine_mapping.h"
#include "pcg/computation_graph.h"
#include "substitutions/sub_parallel_computation_graph.h"

namespace FlexFlow {

struct Strategy {
  ParallelComputationGraph pcg;
  MachineMapping machine_mapping;
  req<float> runtime;
};

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
                   CostEstimator const &cost_estimator,
                   MachineSpecification const &resources,
                   std::function<std::unordered_set<MachineView>(
                       Operator const &, MachineSpecification const &)> const
                       &allowed_machine_views,
                   OptimizerConfig const &opt_config);

} // namespace FlexFlow

VISITABLE_STRUCT(FlexFlow::Strategy, pcg, machine_mapping, runtime);
namespace std {

template <>
struct hash<FlexFlow::Strategy> {
  size_t operator()(FlexFlow::Strategy const &) const;
};

};

#endif

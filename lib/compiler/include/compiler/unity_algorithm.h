#ifndef _FLEXFLOW_COMPILER_UNITY_ALGORITHM_H
#define _FLEXFLOW_COMPILER_UNITY_ALGORITHM_H

#include "cost_estimate.h"
#include "machine_mapping.h"
#include "optimizer_graph.h"

namespace FlexFlow {

struct Substitution {};

struct Strategy {
  OptimizerPCG pcg;
  MachineMapping machine_mapping;

  Strategy(OptimizerPCG const &pcg, MachineMapping const &strategy);
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

Strategy graph_optimize(
    OptimizerComputationGraph &cg,
    ICostEstimator const &cost_estimator,
    MachineSpecification const &resources,
    std::function<std::unordered_set<MachineView>(
        PCGOperatorAttrs const &, MachineSpecification const &)> const
        &allowed_machine_views,
    OptimizerConfig const &opt_config);

}

namespace std {

template <>
struct hash<::FlexFlow::Strategy> {
  size_t operator()(::FlexFlow::Strategy const &) const;
};

}

#endif /* _FLEXFLOW_COMPILER_UNITY_ALGORITHM_H */

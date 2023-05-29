#ifndef _FLEXFLOW_FFC_UNITY_ALGORITHM_H
#define _FLEXFLOW_FFC_UNITY_ALGORITHM_H

#include "op-attrs/operator_attrs.h"
#include "pcg/machine_view.h"
#include "pcg/machine_specification.h"
#include "utils/graph.h"
#include "compiler.h"
#include "cost_estimate.h"

namespace FlexFlow {

using OptimizerComputationGraph =
    NodeLabelledMultiDiGraph<ComputationGraphAttrs>;
using OptimizerPCG =
    LabelledMultiDiGraph<PCGOperatorAttrs, ParallelTensorShape>;

using SubParallelComputationGraph =
    LabelledOpenMultiDiGraph<PCGOperatorAttrs,
                             ParallelTensorShape,
                             MachineView>;

struct Substitution {
};

struct Strategy {
  Strategy(float runtime, std::unordered_map<Node, MachineView> machine_views);
  bool operator<(Strategy const &s) const;

  static Strategy sequential_combine(Strategy const &s1, Strategy const &s2);
  static Strategy parallel_combine(Strategy const &s1, Strategy const &s2);
  static Strategy infinity();

  float runtime;
  std::unordered_map<Node, MachineView> machine_views;
};

Strategy
    optimal_cost(OptimizerPCG const &g,
                 std::function<std::unordered_set<MachineView>(
                     PCGOperatorAttrs const &, MachineSpecification const &)> const
                     &allowed_machine_views,
                 ICostEstimator const &cost_estimator,
                 MachineSpecification const &resources,
                 std::unordered_map<size_t, Strategy> &cached_subgraph_costs);

struct GraphOptResult {
  OptimizerPCG pcg;
  Strategy strategy;

  GraphOptResult(OptimizerPCG const &pcg, Strategy const &strategy);

  bool operator<(GraphOptResult const &r) const;
};

GraphOptResult
    graph_optimize(OptimizerComputationGraph &cg,
                   ICostEstimator const &cost_estimator,
                   MachineSpecification const &resources,
                   std::function<std::unordered_set<MachineView>(
                       PCGOperatorAttrs const &, MachineSpecification const &)> const
                       &allowed_machine_views,
                   OptimizerConfig const &opt_config);
                   
} // namespace FlexFlow

namespace std {
  
template <>
struct hash<::FlexFlow::GraphOptResult> {
  size_t operator()(::FlexFlow::GraphOptResult const &) const;
};

}; // namespace std


#endif

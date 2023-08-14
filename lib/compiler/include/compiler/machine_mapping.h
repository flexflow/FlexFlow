#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_H

#include "cost_estimate.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph.h"
#include "sub_parallel_computation_graph.h"

namespace FlexFlow {

struct MachineMapping {
  static MachineMapping combine(MachineMapping const &, MachineMapping const &);
  static bool nodes_are_disjoint(MachineMapping const &m1,
                                 MachineMapping const &m2);

  req<std::unordered_map<Node, MachineView>> machine_views;
};
FF_VISITABLE_STRUCT(MachineMapping, machine_views);

struct OptimalCostState {
  SerialParallelDecomposition subgraph;
  MachineSpecification resource;
  req<optional<MachineView>> source_machine_view, sink_machine_view;
};
FF_VISITABLE_STRUCT(OptimalCostState,
                    subgraph,
                    resource,
                    source_machine_view,
                    sink_machine_view);

struct OptimalCostResult {
  static OptimalCostResult sequential_combine(OptimalCostResult const &s1,
                                              OptimalCostResult const &s2);
  static OptimalCostResult parallel_combine(OptimalCostResult const &s1,
                                            OptimalCostResult const &s2);
  static OptimalCostResult infinity();

  float runtime;
  MachineMapping machine_mapping;
};
FF_VISITABLE_STRUCT(OptimalCostResult, runtime, machine_mapping);

struct OptimalCostRuntimeCmp {
  bool operator()(OptimalCostResult const &, OptimalCostResult const &);
};

class OptimalCostCache {
public:
  OptimalCostCache() = default;

  optional<OptimalCostResult> load(OptimalCostState const &) const;
  void save(OptimalCostState const &, OptimalCostResult const &);

private:
  std::unordered_map<OptimalCostState, OptimalCostResult> cache;
};

OptimalCostResult
    optimal_cost(ParallelComputationGraph const &g,
                 std::function<std::unordered_set<MachineView>(
                     Operator const &, MachineSpecification const &)> const
                     &allowed_machine_views,
                 CostEstimator const &cost_estimator,
                 MachineSpecification const &resources,
                 OptimalCostCache &cached_subgraph_costs);

} // namespace FlexFlow

#endif

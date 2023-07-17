#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_H

#include "cost_estimate.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph.h"

namespace FlexFlow {

struct MachineMapping {
  static MachineMapping sequential_combine(MachineMapping const &s1,
                                           MachineMapping const &s2);
  static MachineMapping parallel_combine(MachineMapping const &s1,
                                         MachineMapping const &s2);
  static MachineMapping infinity();

  float runtime;
  req<std::unordered_map<Node, MachineView>> machine_views;
};
FF_VISITABLE_STRUCT(MachineMapping, runtime, machine_views);

struct MachineMappingRuntimeCmp {
  bool operator()(MachineMapping const &, MachineMapping const &);
};

MachineMapping optimal_cost(
    ParallelComputationGraph const &g,
    std::function<std::unordered_set<MachineView>(
        Operator const &, MachineSpecification const &)> const
        &allowed_machine_views,
    ICostEstimator const &cost_estimator,
    MachineSpecification const &resources,
    std::unordered_map<size_t, MachineMapping> &cached_subgraph_costs);

} // namespace FlexFlow

#endif

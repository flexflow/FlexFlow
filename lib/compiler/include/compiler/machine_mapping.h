#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_H

#include "cost_estimate.h"
#include "optimizer_graph.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.h"

namespace FlexFlow {

struct MachineMapping : use_visitable_cmp<MachineMapping> {
  MachineMapping(float runtime,
                 std::unordered_map<Node, MachineView> machine_views);

  static MachineMapping sequential_combine(MachineMapping const &s1,
                                           MachineMapping const &s2);
  static MachineMapping parallel_combine(MachineMapping const &s1,
                                         MachineMapping const &s2);
  static MachineMapping infinity();

  float runtime;
  std::unordered_map<Node, MachineView> machine_views;
};

struct MachineMappingRuntimeCmp {
  bool operator()(MachineMapping const &, MachineMapping const &);
};

MachineMapping optimal_cost(
    OptimizerPCG const &g,
    std::function<std::unordered_set<MachineView>(
        PCGOperatorAttrs const &, MachineSpecification const &)> const
        &allowed_machine_views,
    ICostEstimator const &cost_estimator,
    MachineSpecification const &resources,
    std::unordered_map<size_t, MachineMapping> &cached_subgraph_costs);

} // namespace FlexFlow

MAKE_VISIT_HASHABLE(::FlexFlow::MachineMapping);

#endif
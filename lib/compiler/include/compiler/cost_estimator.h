#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_H

#include "cost_estimate.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph.h"
#include "substitutions/sub_parallel_computation_graph.h"

namespace FlexFlow {

float parallel_estimate_cost(
    SubParallelComputationGraphView const &g,
    CostEstimator const &estimator,
    MachineMapping const &device_mapping,
    std::unordered_map<InputMultiDiEdge, MachineView> const
        &frontier_machine_views);

} // namespace FlexFlow

namespace std {

template <>
struct hash<std::unordered_map<FlexFlow::Node, FlexFlow::MachineMapping>> {
  size_t operator()(
      std::unordered_map<FlexFlow::Node, FlexFlow::MachineMapping> const &g)
      const;
};

}; // namespace std

#endif

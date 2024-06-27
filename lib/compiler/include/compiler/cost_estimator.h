#ifndef _FLEXFLOW_COMPILER_COST_ESTIMATOR_H
#define _FLEXFLOW_COMPILER_COST_ESTIMATOR_H

#include "compiler/machine_mapping.h"
#include "cost_estimate.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph.h"
#include "substitutions/sub_parallel_computation_graph.h"

using SubParallelComputationGraphView =
    OutputLabelledOpenMultiDiGraphView<Operator, ParallelTensor>;

namespace FlexFlow {

float parallel_estimate_cost(
    SubParallelComputationGraphView const &g,
    CostEstimator const &estimator,
    MachineMapping const &device_mapping,
    std::unordered_map<InputMultiDiEdge, MachineView> const
        &frontier_machine_views);

} // namespace FlexFlow

#endif

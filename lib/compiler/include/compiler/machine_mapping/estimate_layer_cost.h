#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ESTIMATE_LAYER_COST_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ESTIMATE_LAYER_COST_H

#include "compiler/cost_estimator.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
namespace FlexFlow {

float estimate_layer_cost(ParallelComputationGraph const &pcg,
                          CostEstimator const &cost_estimator,
                          parallel_layer_guid_t const &layer,
                          MachineView const &machine_view);

} // namespace FlexFlow

#endif

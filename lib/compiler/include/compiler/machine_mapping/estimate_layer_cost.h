#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ESTIMATE_LAYER_COST_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ESTIMATE_LAYER_COST_H

#include "compiler/cost_estimator/cost_estimator.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
namespace FlexFlow {

float estimate_layer_cost(CostEstimator const &cost_estimator,
                          UnmappedOpCostEstimateKey const &key,
                          MachineView const &machine_view);

} // namespace FlexFlow

#endif

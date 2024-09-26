#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ESTIMATE_COST_ACROSS_SPLIT_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ESTIMATE_COST_ACROSS_SPLIT_H

#include "compiler/cost_estimator.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.dtg.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"

namespace FlexFlow {

float estimate_cost_across_split(TransitiveReducedPCG const &,
                                 CostEstimator const &,
                                 std::unordered_map<parallel_layer_guid_t, MachineView> const &,
                                 std::unordered_map<parallel_layer_guid_t, MachineView> const &);

} // namespace FlexFlow

#endif

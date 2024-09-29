#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_CONSTRAINTS_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_CONSTRAINTS_H

#include "compiler/machine_mapping/machine_mapping.dtg.h"
#include "compiler/machine_mapping/machine_mapping_context.dtg.h"
#include "compiler/machine_mapping/machine_mapping_constraints.dtg.h"
#include "compiler/machine_mapping/include_unconstrained.dtg.h"
#include "compiler/series_parallel/pcg_binary_sp_decomposition.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"

namespace FlexFlow {

MachineMappingConstraints get_unconstrained_solution_for_layers(std::unordered_set<parallel_layer_guid_t> const &);

std::unordered_set<parallel_layer_guid_t> get_all_layers(MachineMappingConstraints const &, 
                                                         IncludeUnconstrained const &);

std::optional<MachineView> get_machine_view_for_layer(MachineMappingConstraints const &,
                                                      parallel_layer_guid_t const &);

MachineMappingConstraints restrict_domain(MachineMappingConstraints const &, 
                                          std::unordered_set<parallel_layer_guid_t> const &);

MachineMappingConstraints with_additional_constraints(MachineMappingConstraints const &,
                                                      MachineMapping const &);

MachineMapping require_fully_constrained(MachineMappingConstraints const &);

} // namespace FlexFlow

#endif

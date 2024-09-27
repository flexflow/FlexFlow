#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_PARTIAL_MACHINE_MAPPING_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_PARTIAL_MACHINE_MAPPING_H

#include "compiler/machine_mapping/machine_mapping.dtg.h"
#include "compiler/machine_mapping/machine_mapping_context.dtg.h"
#include "compiler/machine_mapping/partial_machine_mapping.dtg.h"
#include "compiler/machine_mapping/include_unconstrained.dtg.h"
#include "compiler/series_parallel/pcg_binary_sp_decomposition.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"

namespace FlexFlow {

PartialMachineMapping get_unconstrained_solution_for_layers(std::unordered_set<parallel_layer_guid_t> const &);

std::unordered_set<parallel_layer_guid_t> get_all_layers(PartialMachineMapping const &, 
                                                         IncludeUnconstrained const &);

std::optional<MachineView> get_machine_view_for_layer(PartialMachineMapping const &,
                                                      parallel_layer_guid_t const &);

PartialMachineMapping get_sub_solution(PartialMachineMapping const &partial_solution, 
                                       PCGBinarySPDecomposition const &sub_problem);

PartialMachineMapping with_additional_layer_machine_views(PartialMachineMapping const &partial_solution,
                                                          std::unordered_map<parallel_layer_guid_t, MachineView> const &additional);

MachineMapping require_complete_mapping(PartialMachineMapping const &);

} // namespace FlexFlow

#endif

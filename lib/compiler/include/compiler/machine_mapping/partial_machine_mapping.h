#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_PARTIAL_MACHINE_MAPPING_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_PARTIAL_MACHINE_MAPPING_H

#include "compiler/machine_mapping/machine_mapping_context.dtg.h"
#include "compiler/machine_mapping/partial_machine_mapping.dtg.h"

namespace FlexFlow {

PartialMachineMapping get_unconstrained_solution();

PartialMachineMapping get_sub_solution(MachineMappingContext const &ctx, 
                                PartialMachineMapping const &partial_solution, 
                                PCGBinarySPDecomposition const &sub_problem);

PartialMachineMapping with_additional_tensor_machine_views(MachineMappingContext const &ctx,
                                                           PartialMachineMapping const &partial_solution,
                                                           std::unordered_map<parallel_tensor_guid_t, MachineView> const &additional);

PartialMachineMapping with_additional_layer_machine_views(MachineMappingContext const &ctx,
                                                          PartialMachineMapping const &partial_solution,
                                                          std::unordered_map<parallel_layer_guid_t, MachineView> const &additional);

} // namespace FlexFlow

#endif

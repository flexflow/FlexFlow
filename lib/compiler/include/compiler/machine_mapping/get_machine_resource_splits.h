#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_GET_MACHINE_RESOURCE_SPLITS_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_GET_MACHINE_RESOURCE_SPLITS_H

#include "pcg/machine_specification.dtg.h"
#include <utility>
#include <unordered_set>

namespace FlexFlow {

std::unordered_set<std::pair<MachineSpecification, MachineSpecification>>
    get_machine_resource_splits(MachineSpecification const &resource);

} // namespace FlexFlow

#endif

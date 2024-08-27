#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_H

#include "machine_mapping.dtg.h"

namespace FlexFlow {

MachineMapping combine(MachineMapping const &, MachineMapping const &);

bool nodes_are_disjoint(MachineMapping const &m1, MachineMapping const &m2);

} // namespace FlexFlow

#endif

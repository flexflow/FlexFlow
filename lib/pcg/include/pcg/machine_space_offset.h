#ifndef _FLEXFLOW_PCG_INCLUDE_MACHINE_SPACE_OFFSET_H
#define _FLEXFLOW_PCG_INCLUDE_MACHINE_SPACE_OFFSET_H

#include "pcg/machine_space_coordinate.dtg.h"
#include "pcg/machine_space_offset.dtg.h"

namespace FlexFlow {

MachineSpaceOffset get_machine_space_offset_from_coordinate(
    MachineSpaceCoordinate const &start, MachineSpaceCoordinate const &coord);

} // namespace FlexFlow

#endif

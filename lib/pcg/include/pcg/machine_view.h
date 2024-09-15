#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_VIEW_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_VIEW_H

#include "machine_specification.dtg.h"
#include "machine_view.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "pcg/task_space_operator.dtg.h"
#include "task_space_coordinate.dtg.h"
#include <cstddef>
#include <unordered_set>

namespace FlexFlow {

MachineSpaceCoordinate
    get_machine_space_coordinate(TaskSpaceOperator const &task,
                                 MachineView const &mv,
                                 TaskSpaceCoordinate const &coordinates,
                                 MachineSpecification const &ms);

std::unordered_set<MachineSpaceCoordinate>
    get_machine_space_coordinates(TaskSpaceOperator const &task,
                                  MachineView const &mv,
                                  MachineSpecification const &ms);

size_t num_dims(MachineView const &mv);

DeviceType get_device_type(MachineView const &mv);

} // namespace FlexFlow

#endif

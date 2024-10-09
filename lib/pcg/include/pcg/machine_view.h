#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_VIEW_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_VIEW_H

#include "machine_specification.dtg.h"
#include "machine_view.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "pcg/operator_task_space.dtg.h"
#include "task_space_coordinate.dtg.h"
#include <cstddef>
#include <optional>
#include <unordered_set>

namespace FlexFlow {

size_t num_dims(MachineView const &mv);

DeviceType get_device_type(MachineView const &mv);

std::vector<stride_t> get_strides(MachineView const &mv);

std::vector<MachineSpecificationDimension>
    get_dimensions(MachineView const &mv);

MachineView machine_view_from_strides_and_machine_spec_dimensions(
    MachineSpaceCoordinate const &start,
    std::vector<stride_t> const &strides,
    std::vector<MachineSpecificationDimension> const &dims);

std::optional<MachineSpaceCoordinate>
    get_machine_space_coordinate(OperatorTaskSpace const &task,
                                 MachineView const &mv,
                                 TaskSpaceCoordinate const &coordinates,
                                 MachineSpecification const &ms);

std::unordered_set<MachineSpaceCoordinate>
    get_machine_space_coordinates(OperatorTaskSpace const &task,
                                  MachineView const &mv,
                                  MachineSpecification const &ms);

} // namespace FlexFlow

#endif

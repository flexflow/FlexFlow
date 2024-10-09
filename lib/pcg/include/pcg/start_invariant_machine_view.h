#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_START_INVARIANT_MACHINE_VIEW_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_START_INVARIANT_MACHINE_VIEW_H

#include "pcg/machine_space_offset.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/operator_task_space.dtg.h"
#include "pcg/start_invariant_machine_view.dtg.h"
#include "pcg/task_space_coordinate.dtg.h"
#include <optional>

namespace FlexFlow {

MachineView
    machine_view_from_start_invariant(StartInvariantMachineView const &mv,
                                      MachineSpaceCoordinate const &start);
StartInvariantMachineView
    start_invariant_from_machine_view(MachineView const &mv);

size_t num_dims(StartInvariantMachineView const &mv);

DeviceType get_device_type(StartInvariantMachineView const &mv);

std::vector<stride_t> get_strides(StartInvariantMachineView const &mv);

std::vector<MachineSpecificationDimension>
    get_dimensions(StartInvariantMachineView const &mv);

StartInvariantMachineView
    start_invariant_machine_view_from_strides_and_machine_spec_dimensions(
        std::vector<stride_t> const &strides,
        std::vector<MachineSpecificationDimension> const &dims);

std::optional<MachineSpaceOffset>
    get_machine_space_offset(OperatorTaskSpace const &task,
                             StartInvariantMachineView const &mv,
                             TaskSpaceCoordinate const &coordinates,
                             MachineSpecification const &ms);

std::unordered_set<MachineSpaceOffset>
    get_machine_space_offsets(OperatorTaskSpace const &task,
                              StartInvariantMachineView const &mv,
                              MachineSpecification const &ms);

} // namespace FlexFlow

#endif

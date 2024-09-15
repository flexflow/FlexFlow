#ifndef _FLEXFLOW_COMPILER_ALLOWED_MACHINE_VIEWS_H
#define _FLEXFLOW_COMPILER_ALLOWED_MACHINE_VIEWS_H

#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/task_space_operator.dtg.h"

namespace FlexFlow {

bool is_valid_machine_view(MachineView const &mv,
                           TaskSpaceOperator const &task,
                           MachineSpecification const &ms);

bool is_valid_machine_view(MachineView const &mv,
                           TaskSpaceOperator const &task);

std::unordered_set<MachineView>
    get_allowed_machine_views(MachineSpecification const &machine_spec,
                              TaskSpaceOperator const &task,
                              DeviceType device_type);

} // namespace FlexFlow

#endif

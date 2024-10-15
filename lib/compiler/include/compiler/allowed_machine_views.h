#ifndef _FLEXFLOW_COMPILER_ALLOWED_MACHINE_VIEWS_H
#define _FLEXFLOW_COMPILER_ALLOWED_MACHINE_VIEWS_H

#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/operator_task_space.dtg.h"

namespace FlexFlow {

bool is_valid_machine_view(MachineView const &mv,
                           OperatorTaskSpace const &task,
                           MachineSpecification const &ms);

std::unordered_set<MachineView>
    get_allowed_machine_views(MachineSpecification const &machine_spec,
                              OperatorTaskSpace const &task,
                              DeviceType device_type);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_START_INVARIANT_MACHINE_VIEW_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_START_INVARIANT_MACHINE_VIEW_H

#include "pcg/machine_view.dtg.h"
#include "pcg/start_invariant_machine_view.dtg.h"

namespace FlexFlow {

MachineView
    machine_view_from_start_invariant(StartInvariantMachineView const &mv,
                                      DeviceCoordinates const &start_id);
StartInvariantMachineView
    start_invariant_from_machine_view(MachineView const &mv);

StartInvariantMachineView make_1d_start_invariant_machine_view(
    num_points_t num_points, stride_t stride, DeviceType device_type);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_START_INVARIANT_MACHINE_VIEW_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_START_INVARIANT_MACHINE_VIEW_H

#include "pcg/machine_view.dtg.h"
#include "pcg/start_invariant_machine_view.dtg.h"

namespace FlexFlow {

MachineView to_start_dependent(StartInvariantMachineView const &mv,
                               device_id_t const &start_id);
StartInvariantMachineView to_start_invariant(MachineView const &mv);

StartInvariantMachineView
    make_1d_start_invariant_machine_view(num_points_t num_points,
                                         stride_t stride);

} // namespace FlexFlow

#endif

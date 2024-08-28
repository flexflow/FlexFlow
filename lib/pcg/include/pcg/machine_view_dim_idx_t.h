#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_VIEW_DIM_IDX_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_VIEW_DIM_IDX_H

#include "pcg/machine_view.dtg.h"
#include "pcg/machine_view_dim_idx_t.dtg.h"

namespace FlexFlow {

std::unordered_set<machine_view_dim_idx_t>
    get_machine_view_indices(MachineView const &mv);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_COMPILER_MACHINE_VIEW_TO_TENSOR_MAPPING_H
#define _FLEXFLOW_COMPILER_MACHINE_VIEW_TO_TENSOR_MAPPING_H

#include "compiler/machine_view_to_tensor_mapping.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "pcg/machine_view.h"

#include <unordered_set>

namespace FlexFlow {

bool is_valid_mapping(MachineViewToTensorMapping const &mapping,
                      MachineView const &mv,
                      ParallelTensorShape const &shape);

std::unordered_set<MachineViewToTensorMapping>
    get_all_machine_view_to_tensor_mappings(MachineView const &mv,
                                            ParallelTensorShape const &shape);

} // namespace FlexFlow

#endif

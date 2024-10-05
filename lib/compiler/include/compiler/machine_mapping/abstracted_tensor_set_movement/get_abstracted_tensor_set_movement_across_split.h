#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_GET_ABSTRACTED_TENSOR_SET_MOVEMENT_ACROSS_SPLIT_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_GET_ABSTRACTED_TENSOR_SET_MOVEMENT_ACROSS_SPLIT_H

#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.dtg.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.dtg.h"
#include "compiler/series_parallel/pcg_binary_series_split.dtg.h"

namespace FlexFlow {

AbstractedTensorSetMovement get_abstracted_tensor_set_movement_across_split(
    TransitiveReducedPCG const &transitive_reduced_pcg,
    PCGBinarySeriesSplit const &split);

} // namespace FlexFlow

#endif

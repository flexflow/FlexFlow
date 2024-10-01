#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ESTIMATE_COST_ACROSS_SPLIT_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ESTIMATE_COST_ACROSS_SPLIT_H

#include "compiler/cost_estimator/tensor_set_movement.dtg.h"
#include "compiler/machine_mapping/parallel_layer_guid_oblivious_machine_mapping.dtg.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.dtg.h"
#include "compiler/series_parallel/pcg_binary_series_split.dtg.h"

namespace FlexFlow {

TensorSetMovement get_tensor_set_movement_across_split(TransitiveReducedPCG const &transitive_reduced_pcg,
                                                       PCGBinarySeriesSplit const &split,
                                                       ParallelLayerGuidObliviousMachineMapping const &pre_mapping,
                                                       ParallelLayerGuidObliviousMachineMapping const &post_mapping);

} // namespace FlexFlow

#endif

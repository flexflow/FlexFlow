#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_PARALLEL_LAYER_GUID_OBLIVIOUS_MACHINE_MAPPING_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_PARALLEL_LAYER_GUID_OBLIVIOUS_MACHINE_MAPPING_H

#include "compiler/machine_mapping/parallel_layer_guid_oblivious_machine_mapping.dtg.h"

namespace FlexFlow {

ParallelLayerGuidObliviousMachineMapping binary_combine_mappings(
    ParallelLayerGuidObliviousMachineMapping const &pre,
    ParallelLayerGuidObliviousMachineMapping const &post);

ParallelLayerGuidObliviousMachineMapping
    restrict_to_left_child(ParallelLayerGuidObliviousMachineMapping const &);
ParallelLayerGuidObliviousMachineMapping
    restrict_to_right_child(ParallelLayerGuidObliviousMachineMapping const &);

std::optional<MachineView>
    get_machine_view_for_path(ParallelLayerGuidObliviousMachineMapping const &,
                              BinaryTreePath const &);

} // namespace FlexFlow

#endif

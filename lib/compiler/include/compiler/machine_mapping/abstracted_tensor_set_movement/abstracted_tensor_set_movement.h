#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ABSTRACTED_TENSOR_SET_MOVEMENT_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ABSTRACTED_TENSOR_SET_MOVEMENT_H

#include "compiler/cost_estimator/tensor_set_movement.dtg.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.dtg.h"
#include "compiler/machine_mapping/machine_mapping.dtg.h"

namespace FlexFlow {

AbstractedTensorSetMovement empty_abstracted_tensor_set_movement();

std::unordered_set<BinaryTreePath> get_src_layers(AbstractedTensorSetMovement const &);
std::unordered_set<BinaryTreePath> get_dst_layers(AbstractedTensorSetMovement const &);

TensorSetMovement concretize_abstracted_tensor_set_movement(AbstractedTensorSetMovement const &,
                                                            MachineMapping const &pre,
                                                            MachineMapping const &post);

} // namespace FlexFlow

#endif

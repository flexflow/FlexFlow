#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.h"
#include "compiler/machine_mapping/parallel_layer_guid_oblivious_machine_mapping.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

AbstractedTensorSetMovement empty_abstracted_tensor_set_movement() {
  return AbstractedTensorSetMovement{{}};
}

std::unordered_set<BinaryTreePath> get_src_layers(AbstractedTensorSetMovement const &m) {
  return flatmap(unordered_set_of(m.single_tensor_movements),
                 [](AbstractedSingleTensorMovement const &s) { 
                   return s.src_machine_views;
                 });
}

std::unordered_set<BinaryTreePath> get_dst_layers(AbstractedTensorSetMovement const &m) {
  return flatmap(unordered_set_of(m.single_tensor_movements),
                 [](AbstractedSingleTensorMovement const &s) { 
                   return s.dst_machine_views;
                 });
}

TensorSetMovement concretize_abstracted_tensor_set_movement(AbstractedTensorSetMovement const &abstracted,
                                                            ParallelLayerGuidObliviousMachineMapping const &pre_mapping,
                                                            ParallelLayerGuidObliviousMachineMapping const &post_mapping) {
  ParallelLayerGuidObliviousMachineMapping mapping = 
    binary_combine_mappings(/*lhs=*/pre_mapping, 
                            /*rhs=*/post_mapping);

  auto concretize_tensor_movement = [&](AbstractedSingleTensorMovement const &a) {
    return SingleTensorMovement{
      /*parallel_tensor_shape=*/a.parallel_tensor_shape,
      /*src_machine_views=*/transform(a.src_machine_views,
                                      [&](BinaryTreePath const &path) {
                                        return get_machine_view_for_path(pre_mapping, path).value();
                                      }),
      /*dst_machine_views=*/transform(a.dst_machine_views,
                                      [&](BinaryTreePath const &path) {
                                        return get_machine_view_for_path(post_mapping, path).value();
                                      }),
    };
  };

  return TensorSetMovement{
    /*single_tensor_movements=*/transform(abstracted.single_tensor_movements, concretize_tensor_movement),
  };
}

} // namespace FlexFlow

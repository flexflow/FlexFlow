#include "compiler/machine_mapping/abstracted_tensor_set_movement.h"
#include "compiler/machine_mapping/partial_machine_mapping.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

AbstractedTensorSetMovement empty_abstracted_tensor_set_movement() {
  return AbstractedTensorSetMovement{{}};
}

std::unordered_set<parallel_layer_guid_t> get_src_layers(AbstractedTensorSetMovement const &m) {
  return flatmap(unordered_set_of(m.single_tensor_movements),
                 [](AbstractedSingleTensorMovement const &s) { 
                   return s.src_machine_views;
                 });
}

std::unordered_set<parallel_layer_guid_t> get_dst_layers(AbstractedTensorSetMovement const &m) {
  return flatmap(unordered_set_of(m.single_tensor_movements),
                 [](AbstractedSingleTensorMovement const &s) { 
                   return s.dst_machine_views;
                 });
}

TensorSetMovement concretize_abstracted_tensor_set_movement(AbstractedTensorSetMovement const &abstracted,
                                                            PartialMachineMapping const &pre_mapping,
                                                            PartialMachineMapping const &post_mapping) {
  auto concretize_tensor_movement = [&](AbstractedSingleTensorMovement const &a) {
    return SingleTensorMovement{
      /*parallel_tensor_shape=*/a.parallel_tensor_shape,
      /*src_machine_views=*/transform(a.src_machine_views,
                                      [&](parallel_layer_guid_t const &layer) {
                                        return get_machine_view_for_layer(pre_mapping, layer).value();
                                      }),
      /*dst_machine_views=*/transform(a.dst_machine_views,
                                      [&](parallel_layer_guid_t const &layer) {
                                        return get_machine_view_for_layer(post_mapping, layer).value();
                                      }),
    };
  };

  return TensorSetMovement{
    /*single_tensor_movements=*/transform(abstracted.single_tensor_movements, concretize_tensor_movement),
  };
}

} // namespace FlexFlow

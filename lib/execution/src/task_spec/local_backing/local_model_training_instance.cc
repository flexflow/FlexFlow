#include "local_model_training_instance.h"
#include "local_allocator.h"

namespace FlexFlow {

void initialize_backing(
    LocalModelTrainingInstance &local_model_training_instance,
    std::unordered_map<OperatorSlotBackingId, GenericTensorAccessorW>
        slot_mapping,
    size_t gpu_memory_size) {
  Allocator allocator = get_local_memory_allocator(gpu_memory_size);

  LocalTrainingBacking local_training_backing(
      local_model_training_instance.computation_graph, allocator, slot_mapping);
  local_model_training_instance.local_training_backing = local_training_backing;
}

GenericTensorAccessorR
    forward(LocalModelTrainingInstance const &local_model_training_instance) {
  LocalTrainingBacking local_training_backing =
      local_model_training_instance.local_training_backing;
  return local_training_backing.execute_forward();
}

void backward(LocalModelTrainingInstance const &local_model_training_instance) {
  LocalTrainingBacking local_training_backing =
      local_model_training_instance.local_training_backing;
  local_training_backing.execute_backward();
}

void update(LocalModelTrainingInstance const &local_model_training_instance) {
  LocalTrainingBacking local_training_backing =
      local_model_training_instance.local_training_backing;
  local_training_backing.execute_update();
}

} // namespace FlexFlow

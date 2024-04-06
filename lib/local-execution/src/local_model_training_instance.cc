#include "local_model_training_instance.h"
#include "local_allocator.h"

namespace FlexFlow {

LocalModelTrainingInstance::LocalModelTrainingInstance(
    ComputationGraph computation_graph,
    Allocator allocator,
    std::unordered_map<tensor_guid_t, GenericTensorAccessorW &> slot_mapping,
    PerDeviceFFHandle handle,
    EnableProfiling enable_profiling,
    ProfilingSettings profiling_settings) {
  this->local_training_backing = LocalTrainingBacking(computation_graph,
                                                      allocator,
                                                      slot_mapping,
                                                      handle,
                                                      enable_profiling,
                                                      profiling_settings);
}

void LocalModelTrainingInstance::forward() {
  this->local_training_backing.execute_forward();
}

void LocalModelTrainingInstance::backward() {
  this->local_training_backing.execute_backward();
}

void LocalModelTrainingInstance::update() {
  this->local_training_backing.execute_update();
}

} // namespace FlexFlow

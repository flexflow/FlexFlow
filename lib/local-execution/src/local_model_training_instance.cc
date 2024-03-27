#include "local_model_training_instance.h"
#include "local_allocator.h"

namespace FlexFlow {

LocalModelTrainingInstance::LocalModelTrainingInstance(
    ComputationGraph graph,
    Allocator allocator,
    Optimizer opt,
    EnableProfiling enable_profiling,
    tensor_guid_t logit_tensor,
    tensor_guid_t label_tensor,
    LossAttrs loss,
    std::unordered_map<OperatorSlotBackingId, GenericTensorAccessorW &> slot_mapping,
    ArgBackingMapping arg_backing_mapping)
  : computation_graph(graph), optimizer(opt), enable_profiling(enable_profiling), logit_tensor(logit_tensor), label_tensor(label_tensor), loss(loss) {
  // this->training_config = TrainingConfig(graph, logit_tensor, label_tensor, loss);
  // this->training_computation_graph = TrainingComputationGraph{graph, opt, enable_profiling};
  this->local_training_backing = LocalTrainingBacking(graph, allocator, slot_mapping, arg_backing_mapping);
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

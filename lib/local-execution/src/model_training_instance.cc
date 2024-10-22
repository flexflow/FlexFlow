#include "local-execution/model_training_instance.h"
#include "pcg/computation_graph.h"
#include "utils/containers/reversed.h"
#include "pcg/optimizer_attrs.h"

namespace FlexFlow {
  
ModelTrainingInstance::ModelTrainingInstance(Allocator const & allocator, 
                                             ComputationGraph const & computation_graph, 
                                             TensorBackingMap const & tensor_backing_map, 
                                             RuntimeArgConfig const & runtime_arg_config, 
                                             LossAttrs const & loss_attrs, 
                                             tensor_guid_t const &logit_tensor, 
                                             tensor_guid_t const &label_tensor, 
                                             OptimizerAttrs const & optimizer_attrs) 
  : computation_graph(computation_graph), training_backing(allocator, computation_graph, tensor_backing_map, runtime_arg_config),
  loss_attrs(loss_attrs), logit_tensor(logit_tensor), label_tensor(label_tensor), optimizer_attrs(optimizer_attrs) {}

void ModelTrainingInstance::register_and_allocate_layers() {
  for (layer_guid_t const & node: topological_ordering(this->computation_graph)) {
    this->training_backing.register_and_allocate_layer(node);
  }
}

void ModelTrainingInstance::allocate_optimizer_tensors() {
  for (layer_guid_t const & node: topological_ordering(this->computation_graph)) {
    this->training_backing.allocate_layer_optimizer_tensors(node, this->optimizer_attrs);
  }
}

void ModelTrainingInstance::execute_init() {
  for (layer_guid_t const & node: topological_ordering(this->computation_graph)) {
    this->training_backing.execute_init(node);
  }
}

PerLayerElapsedTime ModelTrainingInstance::execute_forward() {
  PerLayerElapsedTime per_layer_elapsed_time;
  for (layer_guid_t const & node: topological_ordering(this->computation_graph)) {
    std::optional<float> elapsed_time = this->training_backing.execute_forward(node);
    per_layer_elapsed_time.insert({node, elapsed_time});
  }
  return per_layer_elapsed_time;
}

PerLayerElapsedTime ModelTrainingInstance::execute_backward() {
  this->training_backing.compute_loss(this->loss_attrs, this->logit_tensor, this->label_tensor);
  
  PerLayerElapsedTime per_layer_elapsed_time;
  for (layer_guid_t const & node: reversed(topological_ordering(this->computation_graph))) {
    std::optional<float> elapsed_time = this->training_backing.execute_backward(node);
    per_layer_elapsed_time.insert({node, elapsed_time});
  }
  return per_layer_elapsed_time;
}

void ModelTrainingInstance::execute_update() {
  for (layer_guid_t const & node: topological_ordering(this->computation_graph)) {
    this->training_backing.execute_update(node, this->optimizer_attrs);
  }
  this->optimizer_attrs = get_next_iteration_optimizer_attrs(this->optimizer_attrs);
}

}

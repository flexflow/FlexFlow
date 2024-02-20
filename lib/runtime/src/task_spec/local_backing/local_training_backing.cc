#include "local_training_backing.h"
#include "op_task_invocation.h"
#include "op-attrs/get_task_ids.h"
#include "tasks.h"

namespace FlexFlow {

LocalTrainingBacking::LocalTrainingBacking(ComputationGraph computation_graph, 
                                           Allocator allocator, 
                                           Tensor input_tensor, 
                                           Tensor output_tensor) {
  void * input_ptr = allocator.allocate(input_tensor);
  void * output_ptr = allocator.allocate(output_tensor);

  GenericTensorAccessorR input_tensor_backing = {input_ptr, input_tensor.data_type, input_tensor.dims};
  GenericTensorAccessorR output_tensor_backing = {output_ptr, output_tensor.data_type, output_tensor.dims};

  std::vector<Node> layer_nodes = get_topological_ordering(computation_graph);
  for (Node node: layer_nodes) {
    Layer layer = computation_graph.value().at(node);
    std::vector<task_id_t> task_ids = get_task_ids(layer.attrs);
    for (task_id_t task_id: task_ids) {
      this->task_id_signature_mapping.insert({task_id, get_signature<task_id>()});
      this->task_id_impl_mapping.insert({task_id, get_task_impl<task_id>()});
    }

    // TODO: how to ensure signature and graph edges are synchronized? 
    
    // TODO: allocate tensor for each edge

    this->allocator = allocator;
    this->computation_graph = computation_graph;
    this->input_tensor_backing = input_tensor_backing;
    this->output_tensor_backing = output_tensor_backing;
  }
  not_implemented();
}

GenericTensorAccessorR LocalTrainingBacking::execute_forward() {
  for (auto operator_node: this->topologically_ordered_graph) {
    auto attrs = computation_graph.value().at(operator_node).attrs;
    OpTaskInvocation invocation = forward(operator_node.attrs);
    LocalTaskArgumentAccessor accessor = this->get_fwd_accessor(invocation);
    task_id_impl_mapping[task_id](accessor);
  }
  return this->output_tensor_ptr;
}

void LocalTrainingBacking::execute_backward() {
  for (auto operator_node: std::reverse(this->topologically_ordered_graph.begin(), this->topologically_ordered_graph.end())) {
    auto attrs = computation_graph.value().at(operator_node).attrs;
    OpTaskInvocation invocation = backward(attrs);
    LocalTaskArgumentAccessor accessor = this->get_bwd_accessor(invocation);
    task_id_impl_mapping[task_id](accessor);
  }
}

void LocalTrainingBacking::execute_update() {
  not_implemented();
}


LocalTaskArgumentAccessor LocalTrainingBacking::get_fwd_accessor(OpTaskInvocation invocation) {
  not_implemented();
}

LocalTaskArgumentAccessor LocalTrainingBacking::get_bwd_accessor(OpTaskInvocation invocation) {
  not_implemented();
}


} // namespace FlexFlow

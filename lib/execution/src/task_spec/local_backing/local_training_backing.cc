#include "local_training_backing.h"
#include "local_task_argument_accessor.h"
#include "op-attrs/get_task_ids.h"
#include "op_task_invocation.h"
#include "tasks.h"

namespace FlexFlow {

// pass in map of "which tensors to allocate"
// operator/slot --> GTAR

LocalTrainingBacking::LocalTrainingBacking(
    ComputationGraph const &computation_graph,
    Allocator const &allocator,
    std::unordered_map<OperatorSlotBackingId, GenericTensorAccessorW> const
        &allocated_tensors)
    : op_slot_tensor_mapping(allocated_tensors), allocator(allocator) {
  std::vector<Node> layer_nodes = get_topological_ordering(computation_graph);
  for (Node const &node : layer_nodes) {
    Layer layer = computation_graph.value().at(
        node); // operator_guid_t doesn't seem as expressive as layer -- how to
               // get attrs?
    std::vector<task_id_t> task_ids =
        get_task_ids(layer.attrs); // still think we need this, since we can't
                                   // assume all ops have an init task
    for (task_id_t task_id : task_ids) {
      // TODO: get device state type?
      TaskSignatureImpl<DeviceStateType> task_signature_impl = {get_task_impl<task_id>, get_signature<task_id>};
      this->task_id_mapping.insert({task_id, task_signature_impl});
    }

    // insert tensors
    // incoming edges should already be allocated (either via previously visited
    // nodes or the input map)
    //    TODO: this ^^ should definitely be a test
    std::unordered_set<MultiDiEdge> outgoing_edges =
        get_outgoing_edges(computation_graph, node);
    for (MultiDiEdge const & edge : outgoing_edges) {
      OperatorSlotBackingId src_op_slot = {operator_guid_t(edge.src),
                                           slot_id(edge.src_idx)};
      OperatorSlotBackingId dst_op_slot = {operator_guid_t(edge.dst),
                                           slot_id(edge.dst_idx)};
      auto it = this->op_slot_tensor_mapping.find(src_op_slot);
      if (it != this->op_slot_tensor_mapping.end()) {
        this->op_slot_tensor_mapping.insert({dst_op_slot, it->second});
        continue;
      }

      auto it = this->op_slot_tensor_mapping.find(dst_op_slot);
      if (it != this->op_slot_tensor_mapping.end()) {
        this->op_slot_tensor_mapping.insert({src_op_slot, it->second});
        continue;
      }

      Tensor tensor = computation_graph.value().at(edge);
      void *ptr = this->allocator.allocate(tensor);
      GenericTensorAccessorW tensor_backing = {
          tensor.data_type, tensor.get_shape(), ptr};
      this->op_slot_tensor_mapping.insert({src_op_slot, tensor_backing});
      this->op_slot_tensor_mapping.insert({dst_op_slot, tensor_backing});
    }

    // TODO: register update task
    
    this->topologically_ordered_graph = layer_nodes;
  }
}

// TODO: execute_init
// variant<all device states>

GenericTensorAccessorR LocalTrainingBacking::execute_forward() {
  for (auto operator_node : this->topologically_ordered_graph) {
    auto attrs = computation_graph.value().at(operator_node).attrs;
    OpTaskInvocation invocation = forward(operator_node.attrs);
    LocalTaskArgumentAccessor accessor =
        this->get_task_arg_accessor(invocation);
    task_id_impl_mapping[task_id](accessor);
  }
  return this->output_tensor_ptr;
}

void LocalTrainingBacking::execute_backward() {
  for (auto operator_node :
       std::reverse(this->topologically_ordered_graph.begin(),
                    this->topologically_ordered_graph.end())) {
    auto attrs = computation_graph.value().at(operator_node).attrs;
    OpTaskInvocation invocation = backward(attrs);
    LocalTaskArgumentAccessor accessor =
        this->get_task_arg_accessor(invocation);
    task_id_impl_mapping[task_id](accessor);
  }
}

void LocalTrainingBacking::execute_update() {
  not_implemented();
}

TaskArgumentAccessor
    LocalTrainingBacking::get_task_argument_accessor(OpTaskInvocation invocation) {
  LocalTaskArgumentAccessor local_task_arg_acc(this->allocator);

  OpTaskBinding binding = invocation.binding;
  for (auto tensor_binding : binding.get_tensor_bindings()) {
    std::pair<slot_id, IsGrad> tensor_id = tensor_binding->first;
    operator_guid_t op_guid = invocation.get_operator_guid_t();
    GenericTensorAccessorW tensor_backing =
        this->op_slot_tensor_mapping.at({op_guid, tensor_id->first});
    local_task_arg_acc.insert_tensor(tensor_id, tensor_backing);
  }

  // TODO: do this for args
  binding.get_arg_bindings();
}

} // namespace FlexFlow

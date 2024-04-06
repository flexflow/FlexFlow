#include "local_training_backing.h"
#include "local_task_argument_accessor.h"
#include "get_task_ids.h"
#include "op_task_invocation.h"
#include "utils/exception.h"
#include "tasks.h"

namespace FlexFlow {

TaskRegistry::TaskRegistry(
    std::unordered_map<tensor_guid_t, GenericTensorAccessorW &> 
        allocated_tensors)
    : tensor_mapping(allocated_tensors){};

// void TaskRegistry::register_args(operator_guid_t op, OpArgBacking op_arg_backing) {
//   this->arg_mapping.insert({op, op_arg_backing});
// }

void TaskRegistry::register_task(task_id_t task_id, operator_guid_t op_id) {
  TaskSignatureImpl task_signature_impl = {get_task_impl<task_id>(),
                                           get_signature<task_id>()};
  switch (task_signature_impl.task_signature.type) {
    case OpTaskType::INIT:
      this->init_task_ids.insert({op_id, task_id});
      break;
    case OpTaskType::FWD:
      this->forward_task_ids.insert({op_id, task_id});
      break;
    case OpTaskType::BWD:
      this->backward_task_ids.insert({op_id, task_id});
      break;
    default:
      throw mk_runtime_error("Invalid OpTaskType");
  }
  this->task_mapping.insert({task_id, task_signature_impl});
}

bool TaskRegistry::is_tensor_allocated(tensor_guid_t tensor_id) {
  return this->tensor_mapping.find(tensor_id) != this->tensor_mapping.end();
}

GenericTensorAccessorW & TaskRegistry::get_tensor_backing(tensor_guid_t tensor_id) {
  return this->tensor_mapping.at(tensor_id);
}

// OpArgBacking TaskRegistry::get_arg_backing(operator_guid_t op_id) {
//   return this->arg_mapping.at(op_id);
// }

LocalTrainingBacking::LocalTrainingBacking(
    ComputationGraph computation_graph,
    Allocator allocator,
    std::unordered_map<tensor_guid_t, GenericTensorAccessorW &> slot_mapping,
    PerDeviceFFHandle handle,
    EnableProfiling enable_profiling,
    ProfilingSettings profiling_settings,
    )
    : computation_graph(computation_graph), allocator(allocator), ff_handle(handle), enable_profiling(enable_profiling), profiling_settings(profiling_settings) {
  this->task_registry = TaskRegistry(allocated_tensors);
  std::vector<operator_guid_t> layers = computation_graph.traverse();
  for (operator_guid_t const &node : layers) {
    CompGraphOperatorAttrs attrs = computation_graph.get_layer_attrs(node);
    std::vector<task_id_t> task_ids = get_task_ids(attrs);
    for (task_id_t task_id : task_ids) {
      this->task_registry.register_task(task_id, node);
    }

    // insert tensors
    this->task_registry.input_tensor_slots.insert({node, computation_graph.get_incoming_tensors(node)});
    this->task_registry.output_tensor_slots.insert({node, computation_graph.get_outgoing_tensors(node)});

    for (tensor_guid_t const &edge : outgoing_tensors) {
      if (!this->task_registry.is_tensor_allocated(edge)) {
        const Tensor tensor = computation_graph.at(edge);
        GenericTensorAccessorW tensor_backing =
            this->allocator.allocate(tensor);
        this->task_registry.tensor_mapping.insert({edge, tensor_backing});
      }
    }

  }
  // TODO: register update task

}

void LocalTrainingBacking::execute_init() {
  for (operator_guid_t operator_node :
       get_topological_ordering(this->computation_graph)) {
    auto attrs = computation_graph.value().at(operator_node).attrs;
    OpTaskInvocation invocation = init(attrs);

    assert (validate_invocation(this->task_registry.get_init_signature(operator_node), invocation));
    TaskArgumentAccessor accessor =
        this->get_task_arg_accessor(invocation);
    DeviceSpecific<DeviceStates> device_state = this->call_init_task_impl(invocation.task_id, accessor);
    // this->arg_backing_mapping.at(operator_node).per_device_op_state.second = device_state;
  }
}

DeviceSpecific<DeviceStates> LocalTrainingBacking::call_init_task_impl(task_id_t task_id,
                                          TaskArgumentAccessor acc) {
  TaskSignatureImpl task_sig_impl = this->task_registry.task_mapping.at(task_id);
  auto fn = std::get<std::function<DeviceStates(TaskArgumentAccessor const &)>>(task_sig_impl.impl_function);
  return fn(acc);
}

void LocalTrainingBacking::call_task_impl(task_id_t task_id,
                                          TaskArgumentAccessor acc) {
  TaskSignatureImpl task_sig_impl = this->task_registry.task_mapping.at(task_id);
  auto fn = std::get<std::function<std::optional<float>(TaskArgumentAccessor const &)>>(task_sig_impl.impl_function);
  fn(acc);
}

void LocalTrainingBacking::execute_forward() {
  for (operator_guid_t operator_node : this->computation_graph.traverse()) {
    auto attrs = computation_graph.get_layer_attrs(operator_node);
    OpTaskInvocation invocation = forward(attrs);
    
    assert (validate_invocation(this->task_registry.get_fwd_signature(operator_node), invocation));
    TaskArgumentAccessor accessor =
        this->get_task_arg_accessor(invocation);
    this->call_task_impl(invocation.task_id, accessor);
  }
}

void LocalTrainingBacking::execute_backward() {
  for (operator_guid_t operator_node : this->computation_graph.traverse_reverse_order()) {
    auto attrs = computation_graph.get_layer_attrs(operator_node);
    OpTaskInvocation invocation = backward(attrs);

    assert (validate_invocation(this->task_registry.get_bwd_signature(operator_node), invocation));
    TaskArgumentAccessor accessor =
        this->get_task_arg_accessor(invocation, operator_guid_t);
    this->call_task_impl(invocation.task_id, accessor);
  }
}

void LocalTrainingBacking::execute_update() {
  not_implemented();
}

using SlotGradId = std::pair<slot_id, IsGrad>;

TaskArgumentAccessor
    LocalTrainingBacking::get_task_arg_accessor(OpTaskInvocation invocation, operator_guid_t op_guid) {
  std::unordered_map<SlotGradId, GenericTensorAccessorW &> tensor_backing_map;
  std::unordered_map<slot_id, ArgRefBacking> argument_map;

  OpTaskBinding binding = invocation.binding;
  for (auto tensor_binding : binding.get_tensor_bindings()) {
    SlotGradId slot_grad_id = tensor_binding.first;
    OpTensorSpec tensor_spec = tensor_binding.second;
    std::vector<tensor_guid_t> tensor_slots;
    switch (tensor_spec.role) {
      case TensorRole::INPUT:
        tensor_slots = this->task_registry.input_tensor_slots;
        break;
      case TensorRole::WEIGHT:
        tensor_slots = this->task_registry.weight_tensor_slots;
        break;
      case TensorRole::OUTPUT:
        tensor_slots = this->task_registry.output_tensor_slots;
        break;
      default:
        throw mk_runtime_error("Invalid TensorRole");
    }
    GenericTensorAccessorW tensor_backing =
        this->task_registry.get_tensor_backing(tensor_slots[tensor_spec.idx]);
    tensor_backing_map.insert({slot_grad_id, tensor_backing});
  }

  // OpArgBacking arg_backing = this->arg_backing_mapping.at(op_guid);
  // // TODO: merge maps here
  // // TODO: do this for args
  binding.get_arg_bindings();
  
  return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(
    this->allocator,
    tensor_backing_map,
    argument_map
  );
}

} // namespace FlexFlow

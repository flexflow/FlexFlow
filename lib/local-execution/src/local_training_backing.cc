#include "local_training_backing.h"
#include "utils/exception.h"
#include "get_task_ids.h"
#include "tensor_allocator_utils.h"

namespace FlexFlow {

LocalTrainingBacking::LocalTrainingBacking(
    Allocator allocator,
    ComputationGraph const & computation_graph,
    TensorBackingMapping tensor_backing_mapping,
    RuntimeArgConfig runtime_arg_config)
    :  allocator(allocator), computation_graph(computation_graph) {   // -- err: no default constructor exists for `TaskRegistry`
  this->task_registry = TaskRegistry(tensor_backing_mapping, runtime_arg_config);
  std::vector<operator_guid_t> layers = traverse_comp_graph_forward(computation_graph);
  for (operator_guid_t const &node : layers) {
    CompGraphOperatorAttrs attrs = get_layer_attrs(computation_graph, node);

    // register tasks
    std::vector<task_id_t> task_ids = get_task_ids(attrs);
    for (task_id_t task_id : task_ids) {
      this->task_registry.register_task(task_id, node);
    }

    // insert pre-allocated tensors
    this->task_registry.input_tensor_slots.insert(
        {node, get_incoming_tensors(computation_graph, node)});
    this->task_registry.output_tensor_slots.insert(
        {node, get_outgoing_tensors(computation_graph, node)});

    // allocate new tensors
    for (tensor_guid_t const &edge : get_outgoing_tensors(computation_graph, node)) {
      if (!this->task_registry.is_tensor_allocated(edge)) {
        Tensor const & tensor = computation_graph.at(edge);
        GenericTensorAccessorW tensor_backing = allocate_tensor(this->allocator, tensor);
        this->task_registry.tensor_mapping.insert({edge, tensor_backing});
      }
    }
  }
}

void LocalTrainingBacking::execute_init() {
  for (operator_guid_t const & operator_node :
       traverse_comp_graph_forward(this->computation_graph)) {
    auto attrs = get_layer_attrs(this->computation_graph, operator_node);
    OpTaskInvocation invocation = init(attrs);
    assert(is_invocation_valid(this->task_registry.get_init_signature(operator_node), invocation));

    TaskArgumentAccessor accessor = this->get_task_arg_accessor(invocation, operator_node);
    DeviceSpecific<DeviceStates> device_state =
        this->call_init_task_impl(invocation.task_id, accessor);
    this->task_registry.insert_per_device_op_state(operator_node, device_state);
  }
}

DeviceSpecific<DeviceStates>
    LocalTrainingBacking::call_init_task_impl(task_id_t task_id,
                                              TaskArgumentAccessor acc) {
  TaskSignatureImpl task_sig_impl =
      this->task_registry.task_mapping.at(task_id);
  auto fn = std::get<std::function<DeviceStates(TaskArgumentAccessor const &)>>(
      task_sig_impl.impl_function);
  return fn(acc);
}

void LocalTrainingBacking::call_task_impl(task_id_t task_id,
                                          TaskArgumentAccessor acc) {
  TaskSignatureImpl task_sig_impl =
      this->task_registry.task_mapping.at(task_id);
  auto fn = std::get<
      std::function<std::optional<float>(TaskArgumentAccessor const &)>>(
      task_sig_impl.impl_function);
  fn(acc);
}

void LocalTrainingBacking::execute_forward() {
  for (operator_guid_t operator_node : traverse_comp_graph_forward(this->computation_graph)) {
    auto attrs = get_layer_attrs(this->computation_graph, operator_node);
    OpTaskInvocation invocation = forward(attrs);
    assert(is_invocation_valid(this->task_registry.get_fwd_signature(operator_node), invocation));

    TaskArgumentAccessor accessor = this->get_task_arg_accessor(invocation, operator_node);
    this->call_task_impl(invocation.task_id, accessor);
  }
}

void LocalTrainingBacking::execute_backward() {
  for (operator_guid_t operator_node :
       traverse_comp_graph_backward(computation_graph)) {
    auto attrs = get_layer_attrs(this->computation_graph, operator_node);
    OpTaskInvocation invocation = backward(attrs);

    assert(is_invocation_valid(this->task_registry.get_bwd_signature(operator_node), invocation));
    TaskArgumentAccessor accessor = this->get_task_arg_accessor(invocation, operator_node);
    this->call_task_impl(invocation.task_id, accessor);
  }
}

void LocalTrainingBacking::execute_update() {
  NOT_IMPLEMENTED();
}

TaskArgumentAccessor
    LocalTrainingBacking::get_task_arg_accessor(OpTaskInvocation const & invocation,
                                                operator_guid_t const & op_guid) {
  SlotTensorBackingMapping slot_tensor_backing_map;
  SlotArgBackingMap slot_argument_map;

  this->task_registry.construct_slot_tensor_backing_map(slot_tensor_backing_map, invocation.binding, op_guid);
  this->task_registry.construct_slot_argument_map(slot_argument_map, invocation.binding, op_guid);

  return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(
      this->allocator, slot_tensor_backing_map, slot_argument_map);
}

} // namespace FlexFlow

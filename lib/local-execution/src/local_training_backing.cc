#include "local_training_backing.h"
#include "local_task_argument_accessor.h"
#include "get_task_ids.h"
#include "op_task_invocation.h"
#include "utils/exception.h"
#include "tasks.h"

namespace FlexFlow {

TaskRegistry::TaskRegistry(
    std::unordered_map<OperatorSlotBackingId, GenericTensorAccessorW &> 
        allocated_tensors)
    : tensor_mapping(allocated_tensors){};

void TaskRegistry::register_args(operator_guid_t op, OpArgBacking op_arg_backing) {
  this->arg_mapping.insert({op, op_arg_backing});
}

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

bool TaskRegistry::is_tensor_allocated(OperatorSlotBackingId src_op_slot,
                                       OperatorSlotBackingId dst_op_slot) {
  bool is_allocated = false;

  // if tensor backing exists, then have the dest node point to the same backing
  auto it = this->tensor_mapping.find(src_op_slot);
  if (it != this->tensor_mapping.end()) {
    this->tensor_mapping.insert({dst_op_slot, it->second});
    is_allocated |= true;
  }

  // if tensor backing exists, then have the src node point to the same backing
  it = this->tensor_mapping.find(dst_op_slot);
  if (it != this->tensor_mapping.end()) {
    this->tensor_mapping.insert({src_op_slot, it->second});
    is_allocated |= true;
  }

  return is_allocated;
}

GenericTensorAccessorW & TaskRegistry::get_tensor_backing(OperatorSlotBackingId op_slot_id) {
  return this->tensor_mapping.at(op_slot_id);
}

OpArgBacking TaskRegistry::get_arg_backing(operator_guid_t op_id) {
  return this->arg_mapping.at(op_id);
}

// TODO: switch everything to `operator_guid_t`
LocalTrainingBacking::LocalTrainingBacking(
    ComputationGraph computation_graph,
    Allocator allocator,
    std::unordered_map<OperatorSlotBackingId, GenericTensorAccessorW &> 
        allocated_tensors,
    ArgBackingMapping arg_backing_mapping
    )
    : computation_graph(computation_graph), allocator(allocator), arg_backing_mapping(arg_backing_mapping) {
  this->task_registry = TaskRegistry(allocated_tensors);
  std::vector<Node> layer_nodes = get_topological_ordering(computation_graph);
  for (Node const &node : layer_nodes) {
    Layer layer = computation_graph.value().at(node);
    std::vector<task_id_t> task_ids = get_task_ids(layer.attrs);
    for (task_id_t task_id : task_ids) {
      this->task_registry.register_task(task_id, layer);
    }

    // insert tensors
    // incoming edges should already be allocated (either via previously visited
    // nodes or the input map)
    //    TODO: this ^^ should definitely be a test
    std::unordered_set<MultiDiEdge> outgoing_edges =
        get_outgoing_edges(computation_graph, node);

    for (MultiDiEdge const &edge : outgoing_edges) {
      OperatorSlotBackingId src_op_slot = {operator_guid_t(edge.src),
                                           slot_id(edge.src_idx.value())};
      OperatorSlotBackingId dst_op_slot = {operator_guid_t(edge.dst),
                                           slot_id(edge.dst_idx.value())};
      if (!this->task_registry.is_tensor_allocated(src_op_slot, dst_op_slot)) {
        const Tensor tensor = computation_graph.value().at(edge);
        GenericTensorAccessorW tensor_backing =
            this->allocator.allocate(tensor);
        this->task_registry.tensor_mapping.insert({src_op_slot, tensor_backing});
        this->task_registry.tensor_mapping.insert({dst_op_slot, tensor_backing});
      }
    }

  }
  // TODO: register update task

}

// TODO: execute_init
// variant<all device states>
void LocalTrainingBacking::execute_init() {
  for (operator_guid_t operator_node :
       get_topological_ordering(this->computation_graph)) {
    auto attrs = computation_graph.value().at(operator_node).attrs;
    OpTaskInvocation invocation = init(attrs);

    assert (validate_invocation(this->task_registry.get_init_signature(operator_node), invocation));
    TaskArgumentAccessor accessor =
        this->get_task_arg_accessor(invocation);
    DeviceSpecific<DeviceStates> device_state = this->call_init_task_impl(invocation.task_id, accessor);
    this->arg_backing_mapping.at(operator_node).per_device_op_state.second = device_state;
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

// TODO: don't return GTAR here
void LocalTrainingBacking::execute_forward() {
  for (operator_guid_t operator_node :
       get_topological_ordering(this->computation_graph)) {
    auto attrs = computation_graph.value().at(operator_node).attrs;
    OpTaskInvocation invocation = forward(attrs);
    
    assert (validate_invocation(this->task_registry.get_fwd_signature(operator_node), invocation));
    TaskArgumentAccessor accessor =
        this->get_task_arg_accessor(invocation);
    this->call_task_impl(invocation.task_id, accessor);
  }
}

void LocalTrainingBacking::execute_backward() {
  // containers.h for reversal
  for (operator_guid_t operator_node :
       get_reverse_topological_ordering(this->computation_graph)) {
    auto attrs = computation_graph.value().at(operator_node).attrs;
    OpTaskInvocation invocation = backward(attrs);

    assert (validate_invocation(this->task_registry.get_bwd_signature(operator_node), invocation));
    TaskArgumentAccessor accessor =
        this->get_task_arg_accessor(invocation);
    this->call_task_impl(invocation.task_id, accessor);
  }
}

void LocalTrainingBacking::execute_update() {
  not_implemented();
}

TaskArgumentAccessor
    LocalTrainingBacking::get_task_arg_accessor(OpTaskInvocation invocation) {
  std::unordered_map<SlotGradId, GenericTensorAccessorW> tensor_backing_map;
  std::unordered_map<slot_id, ArgRefBacking> argument_map;

  OpTaskBinding binding = invocation.binding;
  operator_guid_t op_guid = invocation.get_operator_guid_t();
  for (auto tensor_binding : binding.get_tensor_bindings()) {
    std::pair<slot_id, IsGrad> tensor_id = tensor_binding.first;
    OperatorSlotBackingId op_slot_id = {op_guid, tensor_id.first};
    GenericTensorAccessorW tensor_backing =
        this->task_registry.get_tensor_backing(op_slot_id);
    tensor_backing_map.insert({tensor_id, tensor_backing});
  }

  OpArgBacking arg_backing = this->arg_backing_mapping.at(op_guid);
  // TODO: merge maps here
  // TODO: do this for args
  binding.get_arg_bindings();
  
  return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(
    this->allocator,
    tensor_backing_map,
    argument_map
  );
}

} // namespace FlexFlow

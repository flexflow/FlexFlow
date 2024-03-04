#include "local_training_backing.h"
#include "local_task_argument_accessor.h"
#include "op-attrs/get_task_ids.h"
#include "op_task_invocation.h"
#include "tasks.h"

namespace FlexFlow {

TaskRegistry::TaskRegistry(std::unordered_map<OperatorSlotBackingId, GenericTensorAccessorW> const & allocated_tensors)
  : tensor_mapping(allocated_tensors) {};

void TaskRegistry::register_task(task_id_t task_id) {
  TaskSignatureImpl task_signature_impl = {get_task_impl<task_id>, get_signature<task_id>};
  this->task_mapping.insert({task_id, task_signature_impl});
}

bool TaskRegistry::is_tensor_allocated(OperatorSlotBackingId src_op_slot, OperatorSlotBackingId dst_op_slot) {
  bool is_allocated = false;

  // if tensor backing exists, then have the dest node point to the same backing
  auto it = this->tensor_mapping.find(src_op_slot);
  if (it != this->tensor_mapping.end()) {
    this->op_slot_tensor_mapping.insert({dst_op_slot, it->second});
    is_allocated |= true;
  }

  // if tensor backing exists, then have the src node point to the same backing
  auto it = this->tensor_mapping.find(dst_op_slot);
  if (it != this->tensor_mapping.end()) {
    this->tensor_mapping.insert({src_op_slot, it->second});
    is_allocated |= true;
  }

  return is_allocated;
}

void TaskRegistry::get_tensor_backing(OperatorSlotBackingId op_slot_id) {
  return this->tensor_mapping.at(op_slot_id);
}

// TODO: switch everything to `operator_guid_t`
LocalTrainingBacking::LocalTrainingBacking(
    ComputationGraph const &computation_graph,
    Allocator const &allocator,
    std::unordered_map<OperatorSlotBackingId, GenericTensorAccessorW> const
        &allocated_tensors)
    : computation_graph(computation_graph), allocator(allocator) {
  TaskRegistry task_registry (allocated_tensors);
  std::vector<Node> layer_nodes = get_topological_ordering(computation_graph);
  for (Node const &node : layer_nodes) {
    Layer layer = computation_graph.value().at(node);
    std::vector<task_id_t> task_ids = get_task_ids(layer.attrs);
    for (task_id_t task_id : task_ids) {
      this->task_registry.register_task(task_id);
    }

    // insert tensors
    // incoming edges should already be allocated (either via previously visited
    // nodes or the input map)
    //    TODO: this ^^ should definitely be a test
    std::unordered_set<MultiDiEdge> outgoing_edges = get_outgoing_edges(computation_graph, node);

    for (MultiDiEdge const &edge : outgoing_edges) {
      OperatorSlotBackingId src_op_slot = {operator_guid_t(edge.src),
                                            slot_id(edge.src_idx)};
      OperatorSlotBackingId dst_op_slot = {operator_guid_t(edge.dst),
                                            slot_id(edge.dst_idx)};
      if !(task_registry.is_tensor_allocated(src_op_slot, dst_op_slot)) {
        Tensor tensor = computation_graph.value().at(edge);
        GenericTensorAccessorW tensor_backing = this->allocator.allocate(tensor);
        task_registry.tensor_mapping.insert({src_op_slot, tensor_backing});
        task_registry.tensor_mapping.insert({dst_op_slot, tensor_backing});
      }
    }
    

  }
  // TODO: register update task
  
   
  this->task_registry = task_registry;
}

// TODO: execute_init
// variant<all device states>

void LocalTrainingBacking::call_task_impl(task_id_t task_id, TaskArgumentAccessor acc) {
  this->task_registry.task_mapping[task_id](acc);
}

// TODO: don't return GTAR here
void LocalTrainingBacking::execute_forward() {
  for (operator_guid_t operator_node : get_topological_ordering(this->computation_graph)) {
    auto attrs = computation_graph.value().at(operator_node).attrs;
    OpTaskInvocation invocation = forward(attrs);
    // TODO: need to check that OTI complies with OTS
    LocalTaskArgumentAccessor accessor =
        this->get_task_arg_accessor(invocation);
    this->call_task_impl(invocation.task_id, accessor);
  }
}

void LocalTrainingBacking::execute_backward() {
  // containers.h for reversal
  for (operator_guid_t operator_node : get_reverse_topological_ordering(this->computation_graph)) {
    auto attrs = computation_graph.value().at(operator_node).attrs;
    OpTaskInvocation invocation = backward(attrs);
    // TODO: need to check that OTI complies with OTS
    LocalTaskArgumentAccessor accessor =
        this->get_task_arg_accessor(invocation);
    this->call_task_impl(invocation.task_id, accessor);
  }
}

void LocalTrainingBacking::execute_update() {
  not_implemented();
}

TaskArgumentAccessor LocalTrainingBacking::get_task_arg_accessor(
    OpTaskInvocation invocation) {
  std::unordered_map<SlotGradId, GenericTensorAccessorW> tensor_backing_map;

  OpTaskBinding binding = invocation.binding;
  for (auto tensor_binding : binding.get_tensor_bindings()) {
    std::pair<slot_id, IsGrad> tensor_id = tensor_binding->first;
    operator_guid_t op_guid = invocation.get_operator_guid_t();
    OperatorSlotBackingId op_slot_id = {op_guid, tensor_id->first};
    GenericTensorAccessorW tensor_backing = this->task_registry.get_tensor_backing(op_slot_id);
    tensor_backing_map.insert({tensor_id, tensor_backing});
  }
  LocalTaskArgumentAccessor local_task_arg_acc = {this->allocator, tensor_backing_map};

  // TODO: do this for args
  binding.get_arg_bindings();


  return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(local_task_arg_acc);
}

} // namespace FlexFlow

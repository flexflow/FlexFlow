#include "local-execution/local_training_backing.h"
#include "local-execution/loss_functions.h"
#include "local-execution/optimizer.h"
#include "local-execution/task_invocation.h"
#include "local-execution/task_signature_impl.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "utils/containers/contains.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/get_only.h"
#include "utils/exception.h"

namespace FlexFlow {

LocalTrainingBacking::LocalTrainingBacking(
    Allocator const &allocator,
    ComputationGraph const &computation_graph,
    TensorBackingMap const &tensor_backing_mapping,
    RuntimeArgConfig const &runtime_arg_config)
    : allocator(allocator), computation_graph(computation_graph),
      local_slots_backing(tensor_backing_mapping, runtime_arg_config),
      task_registry(empty_task_registry()) {}

void LocalTrainingBacking::register_and_allocate_layer(
    layer_guid_t const &node) {
  ComputationGraphOpAttrs attrs =
      get_layer_attrs(this->computation_graph, node).attrs;
  this->local_slots_backing.allocate_layer_tensors(
      node, this->computation_graph, this->allocator);
  register_tasks_for_layer(this->task_registry, node, attrs);
}

void LocalTrainingBacking::allocate_layer_optimizer_tensors(
    layer_guid_t const &node, OptimizerAttrs const &optimizer_attrs) {
  ComputationGraphOpAttrs attrs =
      get_layer_attrs(this->computation_graph, node).attrs;
  if (attrs.has<WeightAttrs>()) {
    TaskSignature sig = get_update_signature(optimizer_attrs);
    tensor_guid_t weight_tensor =
        get_only(get_outgoing_tensors(this->computation_graph, node));
    this->local_slots_backing.allocate_optimizer_tensors(
        node, weight_tensor, this->computation_graph, this->allocator, sig);
  }
}

DeviceSpecificDeviceStates
    LocalTrainingBacking::call_init_task_impl(task_id_t task_id,
                                              TaskArgumentAccessor const &acc) {
  TaskSignatureAndImpl task_sig_impl =
      this->task_registry.task_mapping.at(task_id);
  auto fn =
      task_sig_impl.impl_function.get<InitOpTaskImplFunction>().function_ptr;
  return fn(acc);
}

std::optional<float>
    LocalTrainingBacking::call_task_impl(task_id_t task_id,
                                         TaskArgumentAccessor acc) {
  TaskSignatureAndImpl task_sig_impl =
      this->task_registry.task_mapping.at(task_id);
  auto fn =
      task_sig_impl.impl_function.get<FwdBwdOpTaskImplFunction>().function_ptr;
  return fn(acc);
}

void LocalTrainingBacking::execute_init(layer_guid_t const &operator_node) {
  if (registry_contains_op_task(
          this->task_registry, operator_node, OpTaskType::INIT)) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(this->computation_graph, operator_node).attrs;

    OpTaskInvocation invocation = init(attrs);
    TaskArgumentAccessor accessor =
        this->get_op_task_arg_accessor(invocation, operator_node);
    DeviceSpecificDeviceStates device_state =
        this->call_init_task_impl(invocation.task_id, accessor);
    this->local_slots_backing.add_per_device_op_state(operator_node,
                                                      device_state);
  }
}

std::optional<float>
    LocalTrainingBacking::execute_forward(layer_guid_t const &operator_node) {
  if (registry_contains_op_task(
          this->task_registry, operator_node, OpTaskType::FWD)) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(this->computation_graph, operator_node).attrs;

    OpTaskInvocation invocation = forward(attrs);
    TaskArgumentAccessor accessor =
        this->get_op_task_arg_accessor(invocation, operator_node);
    return this->call_task_impl(invocation.task_id, accessor);
  } else {
    return std::nullopt;
  }
}

void LocalTrainingBacking::compute_loss(LossAttrs const &loss_attrs,
                                        tensor_guid_t const &logit_tensor,
                                        tensor_guid_t const &label_tensor) {
  assert(this->local_slots_backing.is_tensor_allocated(logit_tensor) &&
         this->local_slots_backing.is_tensor_allocated(label_tensor));
  TaskInvocation loss_invocation =
      backward(loss_attrs, logit_tensor, label_tensor);
  // assert(is_invocation_valid(get_loss_bwd_signature(), loss_invocation));
  TaskArgumentAccessor loss_accessor =
      this->get_task_arg_accessor(loss_invocation);
  TaskImplFunction loss_impl_fn = get_loss_bwd_task_impl();
  loss_impl_fn.get<GenericTaskImplFunction>().function_ptr(loss_accessor);
}

std::optional<float>
    LocalTrainingBacking::execute_backward(layer_guid_t const &operator_node) {
  if (registry_contains_op_task(
          this->task_registry, operator_node, OpTaskType::BWD)) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(this->computation_graph, operator_node).attrs;

    OpTaskInvocation invocation = backward(attrs);
    TaskArgumentAccessor accessor =
        this->get_op_task_arg_accessor(invocation, operator_node);
    return this->call_task_impl(invocation.task_id, accessor);
  } else {
    return std::nullopt;
  }
}

void LocalTrainingBacking::execute_update(
    layer_guid_t const &node, OptimizerAttrs const &optimizer_attrs) {
  LayerAttrs layer_attrs = get_layer_attrs(this->computation_graph, node);
  if (layer_attrs.attrs.has<WeightAttrs>()) {
    // get tensors
    tensor_guid_t weight_tensor =
        get_only(get_outgoing_tensors(this->computation_graph, node));
    std::vector<non_graph_tensor_guid_t> grad_buffer_tensors =
        this->local_slots_backing.weight_optimizer_tensor_guids.at(node);

    // get invocation
    TaskInvocation invocation = get_update_invocation(
        optimizer_attrs, weight_tensor, grad_buffer_tensors);
    // assert(is_invocation_valid(get_update_signature(attrs), invocation));

    // execute update
    TaskArgumentAccessor accessor = this->get_task_arg_accessor(invocation);
    TaskImplFunction update_impl_fn = get_update_task_impl(optimizer_attrs);
    update_impl_fn.get<GenericTaskImplFunction>().function_ptr(accessor);
  }
}

TaskArgumentAccessor LocalTrainingBacking::get_task_arg_accessor(
    TaskInvocation const &invocation) const {
  TensorSlotsBacking tensor_slots_backing =
      this->local_slots_backing.construct_tensor_slots_backing(
          invocation.binding);
  ArgSlotsBacking arg_slots_backing =
      this->local_slots_backing.construct_arg_slots_backing(invocation.binding);
  return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(
      this->allocator, tensor_slots_backing, arg_slots_backing);
}

TaskArgumentAccessor LocalTrainingBacking::get_op_task_arg_accessor(
    OpTaskInvocation const &invocation, layer_guid_t const &op_guid) const {
  TensorSlotsBacking tensor_slots_backing =
      this->local_slots_backing.construct_tensor_slots_backing(
          invocation.binding, op_guid);
  ArgSlotsBacking arg_slots_backing =
      this->local_slots_backing.construct_arg_slots_backing(invocation.binding,
                                                            op_guid);

  return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(
      this->allocator, tensor_slots_backing, arg_slots_backing);
}

void LocalTrainingBacking::insert_tensor(
    tensor_guid_t const &tensor, GenericTensorAccessorW const &tensor_backing) {
  this->local_slots_backing.insert_into_tensor_mapping(tensor, tensor_backing);
}

} // namespace FlexFlow

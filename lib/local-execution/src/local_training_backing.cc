#include "local-execution/local_training_backing.h"
#include "local-execution/loss_functions.h"
#include "local-execution/task_signature_impl.h"
#include "utils/containers/contains.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/reversed.h"
#include "utils/exception.h"

namespace FlexFlow {

LocalTrainingBacking::LocalTrainingBacking(
    Allocator const &allocator,
    ComputationGraph const &computation_graph,
    TensorBackingMap const &tensor_backing_mapping,
    RuntimeArgConfig const &runtime_arg_config,
    std::optional<ModelTrainingInstance> const &training_instance)
    : allocator(allocator), computation_graph(computation_graph),
      local_slots_backing(tensor_backing_mapping, runtime_arg_config),
      task_registry(empty_task_registry()),
      training_instance(training_instance) {

  for (layer_guid_t const &node : topological_ordering(computation_graph)) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(computation_graph, node).attrs;

    // allocate outgoing tensors
    this->local_slots_backing.allocate_outgoing_tensors(
        node, computation_graph, this->allocator);

    // register tasks
    register_tasks_for_layer(this->task_registry, node, attrs);
  }

  if (this->training_instance.has_value()) {
    // label and logit tensor should be allocated
    assert(this->local_slots_backing.is_tensor_allocated(
        this->training_instance.value().label_tensor));
    assert(this->local_slots_backing.is_tensor_allocated(
        this->training_instance.value().logit_tensor));
  }
}

DeviceSpecificDeviceStates
    LocalTrainingBacking::call_init_task_impl(task_id_t task_id,
                                              TaskArgumentAccessor const &acc) {
  TaskSignatureAndImpl task_sig_impl =
      this->task_registry.task_mapping.at(task_id);
  auto fn =
      task_sig_impl.impl_function.get<InitTaskImplFunction>().function_ptr;
  return fn(acc);
}

std::optional<float>
    LocalTrainingBacking::call_task_impl(task_id_t task_id,
                                         TaskArgumentAccessor acc) {
  TaskSignatureAndImpl task_sig_impl =
      this->task_registry.task_mapping.at(task_id);
  auto fn =
      task_sig_impl.impl_function.get<FwdBwdTaskImplFunction>().function_ptr;
  return fn(acc);
}

void LocalTrainingBacking::execute_init() {
  for (layer_guid_t const &operator_node :
       topological_ordering(this->computation_graph)) {
    if (this->task_registry.init_task_ids.at(operator_node).has_value()) {
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
}

PerLayerElapsedTime LocalTrainingBacking::execute_forward() {
  PerLayerElapsedTime per_op_elapsed_time;

  for (layer_guid_t const &operator_node :
       topological_ordering(this->computation_graph)) {
    if (this->task_registry.forward_task_ids.at(operator_node).has_value()) {
      ComputationGraphOpAttrs attrs =
          get_layer_attrs(this->computation_graph, operator_node).attrs;

      OpTaskInvocation invocation = forward(attrs);
      TaskArgumentAccessor accessor =
          this->get_op_task_arg_accessor(invocation, operator_node);
      std::optional<float> elapsed_time =
          this->call_task_impl(invocation.task_id, accessor);
      per_op_elapsed_time.insert({operator_node, elapsed_time});
    }
  }

  return per_op_elapsed_time;
}

PerLayerElapsedTime LocalTrainingBacking::execute_backward() {
  PerLayerElapsedTime per_op_elapsed_time;

  // compute loss
  if (this->training_instance.has_value()) {
    ModelTrainingInstance unwrapped_training_instance =
        training_instance.value();
    TaskInvocation loss_invocation =
        backward(unwrapped_training_instance.loss_attrs,
                 unwrapped_training_instance.logit_tensor,
                 unwrapped_training_instance.label_tensor);
    assert(is_invocation_valid(get_loss_bwd_signature(), loss_invocation));
    TaskArgumentAccessor loss_accessor =
        this->get_task_arg_accessor(loss_invocation);
    TaskImplFunction loss_impl_fn = get_loss_bwd_task_impl();
    loss_impl_fn.get<GenericTaskImplFunction>().function_ptr(loss_accessor);
  }

  // backward through computation graph
  for (layer_guid_t const &operator_node :
       reversed(topological_ordering(this->computation_graph))) {
    if (this->task_registry.backward_task_ids.at(operator_node).has_value()) {
      ComputationGraphOpAttrs attrs =
          get_layer_attrs(this->computation_graph, operator_node).attrs;

      OpTaskInvocation invocation = backward(attrs);
      TaskArgumentAccessor accessor =
          this->get_op_task_arg_accessor(invocation, operator_node);
      std::optional<float> elapsed_time =
          this->call_task_impl(invocation.task_id, accessor);
      per_op_elapsed_time.insert({operator_node, elapsed_time});
    }
  }
  return per_op_elapsed_time;
}

void LocalTrainingBacking::execute_update() {
  NOT_IMPLEMENTED();
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

} // namespace FlexFlow

#include "local-execution/local_training_backing.h"
#include "utils/exception.h"

namespace FlexFlow {

LocalTrainingBacking::LocalTrainingBacking(
    Allocator const &allocator,
    ComputationGraph const &computation_graph,
    TensorBackingMap const &tensor_backing_mapping,
    RuntimeArgConfig const &runtime_arg_config)
    : allocator(allocator), computation_graph(computation_graph),
      local_slots_backing(tensor_backing_mapping, runtime_arg_config) {
  for (layer_guid_t const &node : topological_ordering(computation_graph)) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(computation_graph, node).attrs;

    // allocate outgoing tensors
    this->local_slots_backing.allocate_tensors(
        node, computation_graph, this->allocator);

    // initialize map
    this->task_registry.init_task_ids.insert({node, std::nullopt});
    this->task_registry.forward_task_ids.insert({node, std::nullopt});
    this->task_registry.backward_task_ids.insert({node, std::nullopt});

    if (!(attrs.has<WeightAttrs>() || attrs.has<InputAttrs>() ||
          attrs.has<NoopAttrs>())) {
      // register tasks
      std::vector<task_id_t> task_ids = get_task_ids(attrs);
      for (task_id_t task_id : task_ids) {
        this->task_registry.register_task(task_id, node, attrs);
      }
    }
  }
}

DeviceSpecific<DeviceStates>
    LocalTrainingBacking::call_init_task_impl(task_id_t task_id,
                                              TaskArgumentAccessor const &acc) {
  TaskSignatureAndImpl task_sig_impl =
      this->task_registry.task_mapping.at(task_id);
  auto fn = task_sig_impl.impl_function.get<std::function<
      DeviceSpecific<DeviceStates>(TaskArgumentAccessor const &)>>();
  return fn(acc);
}

std::optional<float>
    LocalTrainingBacking::call_task_impl(task_id_t task_id,
                                         TaskArgumentAccessor acc) {
  TaskSignatureAndImpl task_sig_impl =
      this->task_registry.task_mapping.at(task_id);
  auto fn = task_sig_impl.impl_function.get<
      std::function<std::optional<float>(TaskArgumentAccessor const &)>>();
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
          this->get_task_arg_accessor(invocation, operator_node);
      DeviceSpecific<DeviceStates> device_state =
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
          this->get_task_arg_accessor(invocation, operator_node);
      std::optional<float> elapsed_time =
          this->call_task_impl(invocation.task_id, accessor);
      per_op_elapsed_time.insert({operator_node, elapsed_time});
    }
  }
  return per_op_elapsed_time;
}

PerLayerElapsedTime LocalTrainingBacking::execute_backward() {
  PerLayerElapsedTime per_op_elapsed_time;
  for (layer_guid_t const &operator_node :
       reversed(topological_ordering(this->computation_graph))) {
    if (this->task_registry.backward_task_ids.at(operator_node).has_value()) {
      ComputationGraphOpAttrs attrs =
          get_layer_attrs(this->computation_graph, operator_node).attrs;

      OpTaskInvocation invocation = backward(attrs);
      TaskArgumentAccessor accessor =
          this->get_task_arg_accessor(invocation, operator_node);
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

TaskRegistry const &LocalTrainingBacking::get_task_registry() const {
  return this->task_registry;
}

LocalSlotsBacking const &LocalTrainingBacking::get_local_slots_backing() const {
  return this->local_slots_backing;
}

} // namespace FlexFlow

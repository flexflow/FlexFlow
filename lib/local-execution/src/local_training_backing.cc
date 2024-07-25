#include "local-execution/local_training_backing.h"
#include "utils/exception.h"
#include "utils/containers/reversed.h"

namespace FlexFlow {

LocalTrainingBacking::LocalTrainingBacking(
    Allocator const &allocator,
    ComputationGraph const &computation_graph,
    TensorBackingMap const &tensor_backing_mapping,
    RuntimeArgConfig const &runtime_arg_config)
    : allocator(allocator), computation_graph(computation_graph),
      local_slots_backing(tensor_backing_mapping, runtime_arg_config) {
  std::vector<layer_guid_t> layers = topological_ordering(computation_graph);
  for (layer_guid_t const &node : layers) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(computation_graph, node).attrs;

    // register tasks
    std::vector<task_id_t> task_ids = get_task_ids(attrs);
    for (task_id_t task_id : task_ids) {
      this->task_registry.register_task(task_id, node, attrs);
    }

    // insert pre-allocated tensors
    this->local_slots_backing.input_tensor_slots.insert(
        {node, get_incoming_tensors(computation_graph, node)});
    this->local_slots_backing.output_tensor_slots.insert(
        {node, get_outgoing_tensors(computation_graph, node)});

    // allocate new tensors
    for (tensor_guid_t const &edge :
         get_outgoing_tensors(computation_graph, node)) {
      if (!this->local_slots_backing.is_tensor_allocated(edge)) {
        TensorAttrs tensor_attrs = get_tensor_attrs(computation_graph, edge);
        GenericTensorAccessorW tensor_backing =
            this->allocator.allocate_tensor(tensor_attrs.shape);
        this->local_slots_backing.tensor_mapping.insert({edge, tensor_backing});

        if (tensor_attrs.create_gradients == CreateGrad::YES) {
          GenericTensorAccessorW gradient_tensor_backing =
              this->allocator.allocate_tensor(tensor_attrs.shape);
          this->local_slots_backing.gradient_tensor_mapping.insert(
              {edge, gradient_tensor_backing});
        }
      }
    }
  }
}

DeviceSpecific<DeviceStates>
    LocalTrainingBacking::call_init_task_impl(task_id_t task_id,
                                              TaskArgumentAccessor const &acc) {
  TaskSignatureAndImpl task_sig_impl =
      this->task_registry.task_mapping.at(task_id);
  auto fn = std::get<std::function<DeviceSpecific<DeviceStates>(
      TaskArgumentAccessor const &)>>(task_sig_impl.impl_function);
  return fn(acc);
}

std::optional<float>
    LocalTrainingBacking::call_task_impl(task_id_t task_id,
                                         TaskArgumentAccessor acc) {
  TaskSignatureAndImpl task_sig_impl =
      this->task_registry.task_mapping.at(task_id);
  auto fn = std::get<
      std::function<std::optional<float>(TaskArgumentAccessor const &)>>(
      task_sig_impl.impl_function);
  return fn(acc);
}

void LocalTrainingBacking::execute_init() {
  for (layer_guid_t const &operator_node :
       topological_ordering(this->computation_graph)) {
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

PerLayerElapsedTime LocalTrainingBacking::execute_forward() {
  PerLayerElapsedTime per_op_elapsed_time;
  for (layer_guid_t const &operator_node :
       topological_ordering(this->computation_graph)) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(this->computation_graph, operator_node).attrs;
    OpTaskInvocation invocation = forward(attrs);
    TaskArgumentAccessor accessor =
        this->get_task_arg_accessor(invocation, operator_node);
    std::optional<float> elapsed_time =
        this->call_task_impl(invocation.task_id, accessor);
    per_op_elapsed_time.insert({operator_node, elapsed_time});
  }
  return per_op_elapsed_time;
}

PerLayerElapsedTime LocalTrainingBacking::execute_backward() {
  PerLayerElapsedTime per_op_elapsed_time;
  for (layer_guid_t const &operator_node :
       reversed(topological_ordering(this->computation_graph))) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(this->computation_graph, operator_node).attrs;
    OpTaskInvocation invocation = backward(attrs);
    TaskArgumentAccessor accessor =
        this->get_task_arg_accessor(invocation, operator_node);
    std::optional<float> elapsed_time =
        this->call_task_impl(invocation.task_id, accessor);
    per_op_elapsed_time.insert({operator_node, elapsed_time});
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

} // namespace FlexFlow

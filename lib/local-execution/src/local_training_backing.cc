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

std::optional<float>
    LocalTrainingBacking::execute_kernel(KernelType const &kernel_type) {
  std::optional<float> total_elapsed_time;
  std::vector<layer_guid_t> operators;
  if (kernel_type == KernelType::FWD) {
    operators = topological_ordering(this->computation_graph);
  } else if (kernel_type == KernelType::BWD) {
    operators = reverse_topological_ordering(this->computation_graph);
  } else {
    throw mk_runtime_error("Invalid KernelType, must be FWD or BWD");
  }

  for (layer_guid_t operator_node : operators) {
    auto attrs = get_layer_attrs(this->computation_graph, operator_node).attrs;
    OpTaskInvocation invocation =
        (kernel_type == KernelType::FWD) ? forward(attrs) : backward(attrs);
    TaskArgumentAccessor accessor =
        this->get_task_arg_accessor(invocation, operator_node);
    std::optional<float> elapsed_time =
        this->call_task_impl(invocation.task_id, accessor);

    if (elapsed_time.has_value()) {
      if (total_elapsed_time.has_value()) {
        total_elapsed_time = total_elapsed_time.value() + elapsed_time.value();
      } else {
        total_elapsed_time = elapsed_time.value();
      }
    }
  }
  return total_elapsed_time;
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

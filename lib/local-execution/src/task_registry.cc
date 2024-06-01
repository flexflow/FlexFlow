#include "task_registry.h"

namespace FlexFlow {

TaskRegistry::TaskRegistry(
    TensorBackingMapping allocated_tensors,
    RuntimeArgConfig runtime_arg_config)
    : tensor_mapping(allocated_tensors), runtime_arg_config(runtime_arg_config) {};

void TaskRegistry::register_task(task_id_t task_id, operator_guid_t op_id) {
  // -- err: not a compile-time constant so I can't call the function template I think
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

void TaskRegistry::insert_per_device_op_state(operator_guid_t op_guid, DeviceSpecific<DeviceStates> device_state) {
  this->per_device_op_states.insert({op_guid, device_state});
}

bool TaskRegistry::is_tensor_allocated(tensor_guid_t tensor_id) {
  return this->tensor_mapping.find(tensor_id) != this->tensor_mapping.end();
}

GenericTensorAccessorW const &
    TaskRegistry::get_tensor_backing(tensor_guid_t tensor_id) {
  return this->tensor_mapping.at(tensor_id);
}

void TaskRegistry::construct_slot_tensor_backing_map(
  SlotTensorBackingMapping & mapping,
  OpTaskBinding const & binding,
  operator_guid_t const & op_guid) {
  for (auto tensor_binding : binding.get_tensor_bindings()) {
    SlotGradId slot_grad_id = tensor_binding.first;
    OpTensorSpec tensor_spec = tensor_binding.second;
    std::vector<tensor_guid_t> tensor_guids;
    switch (tensor_spec.role) {
      case TensorRole::INPUT:
        tensor_guids = this->input_tensor_slots.at(op_guid);
        break;
      case TensorRole::WEIGHT:
        tensor_guids = this->weight_tensor_slots.at(op_guid);
        break;
      case TensorRole::OUTPUT:
        tensor_guids = this->output_tensor_slots.at(op_guid);
        break;
      default:
        throw mk_runtime_error("Invalid TensorRole");
    }
    GenericTensorAccessorW tensor_backing =
        this->get_tensor_backing(tensor_guids[tensor_spec.idx]);
    mapping.insert({slot_grad_id, tensor_backing});
  }
}

void TaskRegistry::construct_slot_argument_map(SlotArgBackingMap & mapping, OpTaskBinding const & binding, operator_guid_t const & op_guid) {
  for (auto arg_binding: binding.get_arg_bindings()) {
    slot_id arg_slot = arg_binding.first;
    OpArgSpec op_arg_spec = arg_binding.second;
    if (std::holds_alternative<OpArgRefSpec>(op_arg_spec)) {
      mapping.insert(
        {arg_slot, compile_op_arg_ref_spec(std::get<OpArgRefSpec>(op_arg_spec), op_guid)}
      );
    } else if (std::holds_alternative<RuntimeArgRefSpec>(op_arg_spec), op_guid) {
      mapping.insert(
        {arg_slot, compile_runtime_arg_ref_spec(std::get<RuntimeArgRefSpec>(op_arg_spec))}
      );
    } else if (std::holds_alternative<ConcreteArgSpec>(op_arg_spec)) {
      mapping.insert({arg_slot, std::get<ConcreteArgSpec>(op_arg_spec)});
    } else {
      throw mk_runtime_error("Unhandled argument type");
    }
  }
}

ConcreteArgSpec TaskRegistry::compile_op_arg_ref_spec(OpArgRefSpec op_arg_ref_spec, operator_guid_t const & op_guid) {
  if (op_arg_ref_spec.holds<DeviceSpecific<DeviceStates>>()) {
    return ConcreteArgSpec::create(per_device_op_states.at(op_guid));
  } else if (op_arg_ref_spec.holds<ParallelTensorShape>()) {
    NOT_IMPLEMENTED();  // -- err: see OpArgRefSpec, currently `idx` is not used at all in `input_parallel_tensor_shape(idx)`
  } else {
    throw mk_runtime_error("Unhandled op arg ref type");
  }
}

ConcreteArgSpec TaskRegistry::compile_runtime_arg_ref_spec(RuntimeArgRefSpec runtime_arg_ref_spec) {
  if (runtime_arg_ref_spec.holds<DeviceSpecific<PerDeviceFFHandle>>()) {
    return ConcreteArgSpec::create(this->runtime_arg_config.ff_handle);
  } else if (runtime_arg_ref_spec.holds<EnableProfiling>()) {
    return ConcreteArgSpec::create(this->runtime_arg_config.enable_profiling);
  } else if (runtime_arg_ref_spec.holds<ProfilingSettings>()) {
    return ConcreteArgSpec::create(this->runtime_arg_config.profiling_settings);
  } else {
    throw mk_runtime_error("Unhandled runtime arg ref type");
  }
}


} // namespace FlexFlow
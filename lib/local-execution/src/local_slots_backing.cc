#include "local-execution/local_slots_backing.h"

namespace FlexFlow {

LocalSlotsBacking::LocalSlotsBacking(TensorBackingMap const &allocated_tensors,
                                     RuntimeArgConfig const &runtime_arg_config)
    : tensor_mapping(allocated_tensors),
      runtime_arg_config(runtime_arg_config){};

void LocalSlotsBacking::add_per_device_op_state(
    layer_guid_t const &op_guid,
    DeviceSpecific<DeviceStates> const &device_state) {
  this->per_device_op_states.insert({op_guid, device_state});
}

void LocalSlotsBacking::allocate_new_tensors(
    layer_guid_t const &layer_guid,
    ComputationGraph const &computation_graph,
    Allocator &allocator) {
  std::vector<tensor_guid_t> incoming_tensors =
      get_incoming_tensors(computation_graph, layer_guid);
  std::vector<tensor_guid_t> outgoing_tensors =
      get_outgoing_tensors(computation_graph, layer_guid);
  for (tensor_guid_t const &edge : outgoing_tensors) {
    TensorAttrs tensor_attrs = get_tensor_attrs(computation_graph, edge);
    // tensor allocation
    if (!is_tensor_allocated(edge)) {
      GenericTensorAccessorW tensor_backing =
          allocator.allocate_tensor(tensor_attrs.shape);
      this->tensor_mapping.insert({edge, tensor_backing});
    }

    // gradient tensor allocation
    if (tensor_attrs.create_gradients == CreateGrad::YES && !is_gradient_tensor_allocated(edge)) {
      GenericTensorAccessorW gradient_tensor_backing =
          allocator.allocate_tensor(tensor_attrs.shape);
      this->gradient_tensor_mapping.insert({edge, gradient_tensor_backing});
    }
  }

  this->input_tensor_slots.insert({layer_guid, incoming_tensors});
  this->output_tensor_slots.insert({layer_guid, outgoing_tensors});
}

bool LocalSlotsBacking::is_tensor_allocated(
    tensor_guid_t const &tensor_id) const {
  return contains_key(this->tensor_mapping, tensor_id);
}

bool LocalSlotsBacking::is_gradient_tensor_allocated(
    tensor_guid_t const &tensor_id) const {
  return contains_key(this->gradient_tensor_mapping, tensor_id);
}

GenericTensorAccessorW const &
    LocalSlotsBacking::get_tensor_backing(tensor_guid_t const &tensor_id,
                                          IsGrad is_grad) const {
  switch (is_grad) {
    case IsGrad::NO:
      return this->tensor_mapping.at(tensor_id);
    case IsGrad::YES:
      return this->gradient_tensor_mapping.at(tensor_id);
    default:
      throw mk_runtime_error(fmt::format(
          "IsGrad should only have YES or NO, received {}", is_grad));
  }
}

TensorSlotsBacking LocalSlotsBacking::construct_tensor_slots_backing(
    OpTaskBinding const &binding, layer_guid_t const &op_guid) const {
  TensorSlotsBacking mapping;
  for (auto const &tensor_binding : binding.get_tensor_bindings()) {
    SlotGradId slot_grad_id = tensor_binding.first;
    OpTensorSpec tensor_spec = tensor_binding.second;
    std::vector<tensor_guid_t> tensor_guids;
    switch (tensor_spec.role) {
      case TensorRole::INPUT:
      case TensorRole::WEIGHT:
        tensor_guids = this->input_tensor_slots.at(op_guid);
        break;
      case TensorRole::OUTPUT: 
        tensor_guids = this->output_tensor_slots.at(op_guid);
        break;
      default:
        throw mk_runtime_error(
            fmt::format("Invalid TensorRole")); // inserting role yields
                                                // "type_is_unformattable" error
    }
    IsGrad is_grad = slot_grad_id.second;
    
    GenericTensorAccessorW tensor_backing =
        this->get_tensor_backing(tensor_guids.at(tensor_spec.idx), is_grad);
    mapping.insert({slot_grad_id, tensor_backing});
  }
  return mapping;
}

ArgSlotsBacking LocalSlotsBacking::construct_arg_slots_backing(
    OpTaskBinding const &binding, layer_guid_t const &op_guid) const {
  ArgSlotsBacking mapping;
  for (auto const &arg_binding : binding.get_arg_bindings()) {
    slot_id arg_slot = arg_binding.first;
    OpArgSpec op_arg_spec = arg_binding.second;
    if (std::holds_alternative<OpArgRefSpec>(op_arg_spec)) {
      mapping.insert({arg_slot,
                      resolve_op_arg_ref_spec(
                          std::get<OpArgRefSpec>(op_arg_spec), op_guid)});
    } else if (std::holds_alternative<RuntimeArgRefSpec>(op_arg_spec)) {
      mapping.insert({arg_slot,
                      resolve_runtime_arg_ref_spec(
                          std::get<RuntimeArgRefSpec>(op_arg_spec))});
    } else if (std::holds_alternative<ConcreteArgSpec>(op_arg_spec)) {
      mapping.insert({arg_slot, std::get<ConcreteArgSpec>(op_arg_spec)});
    } else {
      throw mk_runtime_error("Unhandled argument type");
    }
  }
  return mapping;
}

ConcreteArgSpec LocalSlotsBacking::resolve_op_arg_ref_spec(
    OpArgRefSpec const &op_arg_ref_spec, layer_guid_t const &op_guid) const {
  if (op_arg_ref_spec.holds<DeviceSpecific<DeviceStates>>()) {
    return ConcreteArgSpec::create(per_device_op_states.at(op_guid));
  } else if (op_arg_ref_spec.holds<ParallelTensorShape>()) {
    ParallelTensorShapeRefType index_op_arg_ref =
        std::get<ParallelTensorShapeRefType>(op_arg_ref_spec.get_ref_type());
    std::vector<tensor_guid_t> input_tensor_guids =
        this->input_tensor_slots.at(op_guid);
    GenericTensorAccessorW tensor_backing = this->get_tensor_backing(
        input_tensor_guids.at(index_op_arg_ref.idx), IsGrad::NO);
    ParallelTensorShape shape = lift_to_parallel(
        get_tensor_shape(tensor_backing.shape, tensor_backing.data_type));
    return ConcreteArgSpec::create(shape);
  } else {
    throw mk_runtime_error("Unhandled op arg ref type");
  }
}

ConcreteArgSpec LocalSlotsBacking::resolve_runtime_arg_ref_spec(
    RuntimeArgRefSpec const &runtime_arg_ref_spec) const {
  if (runtime_arg_ref_spec.holds<DeviceSpecific<PerDeviceFFHandle>>()) {
    return ConcreteArgSpec::create(this->runtime_arg_config.ff_handle);
  } else if (runtime_arg_ref_spec.holds<ProfilingSettings>()) {
    return ConcreteArgSpec::create(this->runtime_arg_config.profiling_settings);
  } else {
    throw mk_runtime_error("Unhandled runtime arg ref type");
  }
}

bool are_maps_virtually_equivalent(TensorBackingMap const &map1,
                                   TensorBackingMap const &map2) {
  if (map1.size() != map2.size()) {
    return false;
  }

  for (std::pair<tensor_guid_t, GenericTensorAccessorW> const &pairing : map1) {
    if (!contains_key(map2, pairing.first)) {
      return false;
    }
    GenericTensorAccessorW acc2 = map2.at(pairing.first);
    if (!is_shape_and_dtype_equal(pairing.second, acc2)) {
      return false;
    }
  }
  return true;
}

} // namespace FlexFlow

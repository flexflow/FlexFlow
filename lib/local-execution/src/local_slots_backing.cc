#include "local-execution/local_slots_backing.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/computation_graph.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/map_values.h"
#include "utils/overload.h"

namespace FlexFlow {

LocalSlotsBacking::LocalSlotsBacking(TensorBackingMap const &allocated_tensors,
                                     RuntimeArgConfig const &runtime_arg_config)
    : tensor_mapping(allocated_tensors),
      runtime_arg_config(runtime_arg_config){};

void LocalSlotsBacking::add_per_device_op_state(
    layer_guid_t const &op_guid,
    DeviceSpecificDeviceStates const &device_state) {
  this->per_device_op_states.insert({op_guid, device_state});
}

void LocalSlotsBacking::allocate_outgoing_tensors(
    layer_guid_t const &layer_guid,
    ComputationGraph const &computation_graph,
    Allocator &allocator) {
  std::vector<tensor_guid_t> incoming_input_tensors =
      get_incoming_inputs(computation_graph, layer_guid);
  std::vector<tensor_guid_t> incoming_weight_tensors =
      get_incoming_weights(computation_graph, layer_guid);
  std::vector<tensor_guid_t> outgoing_tensors =
      get_outgoing_tensors(computation_graph, layer_guid);
  for (tensor_guid_t const &output_tensor : outgoing_tensors) {
    TensorAttrs tensor_attrs =
        get_tensor_attrs(computation_graph, output_tensor);
    // tensor allocation
    if (!is_tensor_allocated(output_tensor)) {
      GenericTensorAccessorW tensor_backing =
          allocator.allocate_tensor(tensor_attrs.shape);
      this->tensor_mapping.insert({output_tensor, tensor_backing});
    }

    // gradient tensor allocation
    if (tensor_attrs.create_gradients == CreateGrad::YES &&
        !is_gradient_tensor_allocated(output_tensor)) {
      GenericTensorAccessorW gradient_tensor_backing =
          allocator.allocate_tensor(tensor_attrs.shape);
      this->gradient_tensor_mapping.insert(
          {output_tensor, gradient_tensor_backing});
    }
  }

  this->input_tensor_slots.insert({layer_guid, incoming_input_tensors});
  this->weight_tensor_slots.insert({layer_guid, incoming_weight_tensors});
  this->output_tensor_slots.insert({layer_guid, outgoing_tensors});
}

void LocalSlotsBacking::allocate_optimizer_tensors(
    layer_guid_t const &weight_layer,
    tensor_guid_t const &weight,
    ComputationGraph const &cg,
    Allocator &allocator,
    TaskSignature const &sig) {
  GenericTensorAccessorW weight_backing =
      get_tensor_backing(weight, IsGrad::NO);
  int num_grad_buffer_tensors =
      sig.tensor_guid_slots.size() - 2; // ignore 2 (weight and weight_grad)
  std::vector<tensor_guid_t> grad_buffer_tensors =
      get_new_tensor_guids_for_layer_without_graph_insertion(
          cg, weight_layer, num_grad_buffer_tensors);
  for (tensor_guid_t const &tensor_guid : grad_buffer_tensors) {
    GenericTensorAccessorW buffer_backing = allocator.allocate_tensor(
        get_tensor_shape(weight_backing.shape, weight_backing.data_type));
    this->gradient_tensor_mapping.insert({tensor_guid, buffer_backing});
  }
  this->weight_optimizer_tensor_guids.insert({weight, grad_buffer_tensors});
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
      assert(contains_key(this->tensor_mapping, tensor_id));
      return this->tensor_mapping.at(tensor_id);
    case IsGrad::YES:
      assert(contains_key(this->gradient_tensor_mapping, tensor_id));
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
    int weight_adjusted_idx = 0;
    switch (tensor_spec.role) {
      case TensorRole::WEIGHT:
        assert(contains_key(this->weight_tensor_slots, op_guid));
        tensor_guids = this->weight_tensor_slots.at(op_guid);
        break;
      case TensorRole::INPUT:
        assert(contains_key(this->input_tensor_slots, op_guid));
        tensor_guids = this->input_tensor_slots.at(op_guid);
        break;
      case TensorRole::OUTPUT:
        assert(contains_key(this->output_tensor_slots, op_guid));
        tensor_guids = this->output_tensor_slots.at(op_guid);
        break;
      default:
        throw mk_runtime_error(
            fmt::format("Invalid TensorRole {}", tensor_spec.role));
    }

    IsGrad is_grad = slot_grad_id.is_grad;
    GenericTensorAccessorW tensor_backing =
        this->get_tensor_backing(tensor_guids.at(tensor_spec.idx), is_grad);

    mapping.insert({slot_grad_id, tensor_backing});
  }
  return mapping;
}

TensorSlotsBacking LocalSlotsBacking::construct_tensor_slots_backing(
    TaskBinding const &binding) const {
  TensorSlotsBacking mapping;

  for (auto const &tensor_binding : binding.get_tensor_bindings()) {
    SlotGradId slot_grad_id = tensor_binding.first;
    TensorGuidSpec tensor_spec = tensor_binding.second;

    GenericTensorAccessorW accessor =
        this->get_tensor_backing(tensor_spec.tensor_guid, slot_grad_id.is_grad);
    mapping.insert({slot_grad_id, accessor});
  }

  return mapping;
}

ArgSlotsBacking LocalSlotsBacking::construct_arg_slots_backing(
    OpTaskBinding const &binding, layer_guid_t const &op_guid) const {
  return map_values(
      binding.get_arg_bindings(), [&](OpArgSpec const &arg_binding) {
        return arg_binding.template visit<ConcreteArgSpec>(
            overload{[&](OpArgRefSpec const &s) {
                       return this->resolve_op_arg_ref_spec(s, op_guid);
                     },
                     [&](RuntimeArgRefSpec const &s) {
                       return this->resolve_runtime_arg_ref_spec(s);
                     },
                     [](ConcreteArgSpec const &s) { return s; }});
      });
}

ArgSlotsBacking LocalSlotsBacking::construct_arg_slots_backing(
    TaskBinding const &binding) const {
  return map_values(
      binding.get_arg_bindings(), [&](TaskArgSpec const &arg_binding) {
        return arg_binding.template visit<ConcreteArgSpec>(
            overload{[&](RuntimeArgRefSpec const &s) {
                       return this->resolve_runtime_arg_ref_spec(s);
                     },
                     [](ConcreteArgSpec const &s) { return s; }});
      });
  ;
}

ConcreteArgSpec LocalSlotsBacking::resolve_op_arg_ref_spec(
    OpArgRefSpec const &op_arg_ref_spec, layer_guid_t const &op_guid) const {
  if (op_arg_ref_spec.holds<DeviceSpecificDeviceStates>()) {
    assert(contains_key(per_device_op_states, op_guid));
    DeviceSpecificDeviceStates device_specific =
        per_device_op_states.at(op_guid);
    PerDeviceOpState device_state =
        get_device_state_from_device_specific(device_specific, 0);
    return ConcreteArgSpec::create(device_state);
  } else if (op_arg_ref_spec.holds<ParallelTensorShape>()) {
    ParallelTensorShapeRefType index_op_arg_ref =
        op_arg_ref_spec.get_ref_type().get<ParallelTensorShapeRefType>();

    assert(contains_key(this->input_tensor_slots, op_guid));
    std::vector<tensor_guid_t> input_tensor_guids =
        this->input_tensor_slots.at(op_guid);

    assert(input_tensor_guids.size() > index_op_arg_ref.idx);
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
    return ConcreteArgSpec::create(
        *(this->runtime_arg_config.ff_handle.get(0)));
  } else if (runtime_arg_ref_spec.holds<ProfilingSettings>()) {
    return ConcreteArgSpec::create(this->runtime_arg_config.profiling_settings);
  } else {
    throw mk_runtime_error("Unhandled runtime arg ref type");
  }
}

} // namespace FlexFlow

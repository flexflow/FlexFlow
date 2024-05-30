#include "op_task_invocation.h"

namespace FlexFlow {

OpTensorSpec input_tensor(int idx,
                          OpSlotOptions option = OpSlotOptions::NECESSARY) {
  return {TensorRole::INPUT, option, idx};
}

OpTensorSpec output_tensor(int idx,
                           OpSlotOptions option = OpSlotOptions::NECESSARY) {
  return {TensorRole::OUTPUT, option, idx};
}

OpTensorSpec weight_tensor(int idx,
                           OpSlotOptions option = OpSlotOptions::NECESSARY) {
  return {TensorRole::WEIGHT, option, idx};
}

void OpTaskBinding::bind(slot_id slot, OpTensorSpec const &tensor_spec) {
  this->tensor_bindings.insert({{slot, IsGrad::NO}, tensor_spec});
}

void OpTaskBinding::bind_grad(slot_id slot, OpTensorSpec const &tensor_spec) {
  this->tensor_bindings.insert({{slot, IsGrad::YES}, tensor_spec});
}

std::unordered_map<std::pair<slot_id, IsGrad>, OpTensorSpec> const &
    OpTaskBinding::get_tensor_bindings() const {
  return this->tensor_bindings;
}

std::unordered_map<slot_id, OpArgSpec> const &
    OpTaskBinding::get_arg_bindings() const {
  return this->arg_bindings;
}

OpTaskBinding infer_bwd_binding(OpTaskBinding const &fwd) {
  OpTaskBinding bwd;
  bwd.arg_bindings = fwd.get_arg_bindings();
  bwd.tensor_bindings = fwd.get_tensor_bindings();
  for (auto const &[key, spec] : fwd.get_tensor_bindings()) {
    OpSlotOptions slot_option = spec.slot_option;
    if (slot_option != OpSlotOptions::UNTRAINABLE ||
        slot_option != OpSlotOptions::OPTIONAL_UNTRAINABLE) {
      slot_id slot = key.first;
      bwd.bind_grad(slot, spec);
    }
  }
  return bwd;
}

bool is_op_tensor_spec_invalid(OpTensorSlotSpec tensor_slot_spec,
                               OpTensorSpec tensor_spec) {
  return tensor_spec.role != tensor_slot_spec.tensor_role ||
         tensor_spec.slot_option != tensor_slot_spec.slot_option;
}

bool is_tensor_invocation_valid(OpTaskSignature const &sig,
                                OpTaskInvocation const &inv) {
  auto tensor_bindings = inv.binding.get_tensor_bindings();
  for (OpTensorSlotSpec const &op_tensor_slot_spec : sig.get_tensor_slots()) {
    std::pair<slot_id, IsGrad> tensor_key =
        std::make_pair(op_tensor_slot_spec.name, op_tensor_slot_spec.is_grad);
    OpTensorSpec const &op_tensor_spec = tensor_bindings.at(tensor_key);
    if (is_op_tensor_spec_invalid(op_tensor_slot_spec, op_tensor_spec)) {
      return false;
    }
  }
  return true;
}

bool is_arg_type_invalid(std::type_index expected_arg_type,
                         OpArgSpec op_arg_spec) {
  std::type_index arg_spec_type = std::visit(
      [](auto &&arg) -> std::type_index { return arg.get_type_index(); },
      op_arg_spec);
  return arg_spec_type != expected_arg_type;
}

bool is_arg_invocation_valid(OpTaskSignature const &sig,
                             OpTaskInvocation const &inv) {
  auto sig_arg_types = sig.get_arg_types();
  for (auto arg_binding : inv.binding.get_arg_bindings()) {
    std::type_index arg_type = sig_arg_types.at(arg_binding.first);
    if (is_arg_type_invalid(arg_type, arg_binding.second)) {
      return false;
    }
  }

  return true;
}

bool is_invocation_valid(OpTaskSignature const &sig,
                         OpTaskInvocation const &inv) {
  return is_tensor_invocation_valid(sig, inv) &&
         is_arg_invocation_valid(sig, inv);
}

} // namespace FlexFlow

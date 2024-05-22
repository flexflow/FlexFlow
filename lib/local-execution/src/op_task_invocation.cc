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
  bwd.bind_args_from_fwd(fwd);
  bwd.bind_tensors_from_fwd(fwd);
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

bool validate_invocation(OpTaskSignature sig, OpTaskInvocation inv) {
  // tensors
  auto tensor_bindings = inv.binding.get_tensor_bindings();
  for (OpTensorSlotSpec const &op_tensor_slot_spec : sig.get_tensor_slots()) {
    slot_id name = op_tensor_slot_spec.name;
    IsGrad is_grad = op_tensor_slot_spec.is_grad;
    std::pair<slot_id, IsGrad> tensor_key = std::make_pair(name, is_grad);
    OpTensorSpec const &op_tensor_spec = tensor_bindings.at(tensor_key);
    if (op_tensor_spec.role != op_tensor_slot_spec.tensor_role ||
        op_tensor_spec.slot_option != op_tensor_slot_spec.slot_option) {
      return false;
    }
  }

  // args
  auto sig_arg_types = sig.get_arg_types();
  OpArgSpecTypeAccessor type_accessor;
  for (auto arg_binding : inv.binding.get_arg_bindings()) {
    slot_id name = arg_binding.first;
    OpArgSpec op_arg_spec = arg_binding.second;
    std::type_index arg_type = sig_arg_types.at(name);
    if (type_accessor(op_arg_spec) != arg_type) {
      return false;
    }
  }

  return true;
}

} // namespace FlexFlow

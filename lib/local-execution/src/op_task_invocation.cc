#include "local-execution/op_task_invocation.h"
#include "local-execution/op_arg_spec.h"
#include "utils/containers/contains_key.h"

namespace FlexFlow {

void OpTaskBinding::bind(
    int slot, VariadicTensorRef<OpTensorSpec> const &variadic_tensor_ref) {
  this->bind(slot_id_t{slot}, variadic_tensor_ref);
}

void OpTaskBinding::bind(
    slot_id_t slot,
    VariadicTensorRef<OpTensorSpec> const &variadic_tensor_ref) {
  NOT_IMPLEMENTED();
}

void OpTaskBinding::bind(int slot, OpTensorSpec const &tensor_spec) {
  this->bind(slot_id_t{slot}, tensor_spec);
}

void OpTaskBinding::bind(slot_id_t slot, OpTensorSpec const &tensor_spec) {
  this->tensor_bindings.insert({SlotGradId{slot, IsGrad::NO}, tensor_spec});
}

void OpTaskBinding::bind_grad(int slot, OpTensorSpec const &tensor_spec) {
  this->bind_grad(slot_id_t{slot}, tensor_spec);
}

void OpTaskBinding::bind_grad(slot_id_t slot, OpTensorSpec const &tensor_spec) {
  this->tensor_bindings.insert({SlotGradId{slot, IsGrad::YES}, tensor_spec});
}

void OpTaskBinding::insert_arg_spec(slot_id_t name, OpArgSpec const &arg_spec) {
  assert(!contains_key(this->arg_bindings, name));
  this->arg_bindings.insert({name, arg_spec});
}

bool OpTaskBinding::operator==(OpTaskBinding const &other) const {
  return this->tie() == other.tie();
}

bool OpTaskBinding::operator!=(OpTaskBinding const &other) const {
  return this->tie() != other.tie();
}

std::tuple<std::unordered_map<SlotGradId, OpTensorSpec> const &,
           std::unordered_map<slot_id_t, OpArgSpec> const &>
    OpTaskBinding::tie() const {
  return std::tie(this->tensor_bindings, this->arg_bindings);
}

std::unordered_map<SlotGradId, OpTensorSpec> const &
    OpTaskBinding::get_tensor_bindings() const {
  return this->tensor_bindings;
}

std::unordered_map<slot_id_t, OpArgSpec> const &
    OpTaskBinding::get_arg_bindings() const {
  return this->arg_bindings;
}

void OpTaskBinding::bind_from_forward(OpTaskBinding const &fwd) {
  this->arg_bindings = fwd.get_arg_bindings();
  this->tensor_bindings = fwd.get_tensor_bindings();
}

OpTaskBinding infer_bwd_binding(OpTaskBinding const &fwd) {
  OpTaskBinding bwd;
  bwd.bind_from_forward(fwd);
  for (auto const &[key, spec] : fwd.get_tensor_bindings()) {
    OpSlotOptions slot_option = spec.slot_option;
    if (slot_option != OpSlotOptions::UNTRAINABLE ||
        slot_option != OpSlotOptions::OPTIONAL_UNTRAINABLE) {
      slot_id_t slot = key.slot_id;
      bwd.bind_grad(slot, spec);
    }
  }
  return bwd;
}

bool is_op_tensor_spec_invalid(OpTensorSlotSpec const &tensor_slot_spec,
                               OpTensorSpec const &tensor_spec) {
  return tensor_spec.role != tensor_slot_spec.tensor_role ||
         tensor_spec.slot_option != tensor_slot_spec.slot_option;
}

bool is_tensor_invocation_valid(OpTaskSignature const &sig,
                                OpTaskInvocation const &inv) {
  auto tensor_bindings = inv.binding.get_tensor_bindings();
  for (OpTensorSlotSpec const &op_tensor_slot_spec : sig.get_tensor_slots()) {
    SlotGradId tensor_key =
        SlotGradId{op_tensor_slot_spec.name, op_tensor_slot_spec.is_grad};
    OpTensorSpec op_tensor_spec = tensor_bindings.at(tensor_key);
    if (is_op_tensor_spec_invalid(op_tensor_slot_spec, op_tensor_spec)) {
      return false;
    }
  }

  // FIXME -- make sure invocation doesn't contain MORE than signature
  // https://github.com/flexflow/FlexFlow/issues/1442
  return true;
}

bool is_arg_type_invalid(std::type_index expected_arg_type,
                         OpArgSpec op_arg_spec) {
  std::type_index arg_spec_type = get_op_arg_spec_type_index(op_arg_spec);
  return arg_spec_type != expected_arg_type;
}

bool is_arg_invocation_valid(OpTaskSignature const &sig,
                             OpTaskInvocation const &inv) {
  // FIXME -- arg signature/invocation checking
  // https://github.com/flexflow/FlexFlow/issues/1442
  // auto sig_arg_types = sig.get_arg_types();
  // for (auto arg_binding : inv.binding.get_arg_bindings()) {
  //   std::type_index arg_type = sig_arg_types.at(arg_binding.first);
  //   assert (!is_arg_type_invalid(arg_type, arg_binding.second));
  // }

  return true;
}

bool is_invocation_valid(OpTaskSignature const &sig,
                         OpTaskInvocation const &inv) {
  return is_tensor_invocation_valid(sig, inv) &&
         is_arg_invocation_valid(sig, inv);
}

} // namespace FlexFlow

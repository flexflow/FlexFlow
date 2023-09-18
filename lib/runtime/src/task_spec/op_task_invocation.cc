#include "op_task_invocation.h"
#include "task_argument_accessor.h"

namespace FlexFlow {

OpTensorSpec input_tensor(int idx, OpSlotOptions option = OpSlotOptions::NECESSARY) {
  return {TensorRole::INPUT, option, idx};
}

OpTensorSpec output_tensor(int idx, OpSlotOptions option = OpSlotOptions::NECESSARY) {
  return {TensorRole::OUTPUT, option, idx};
}

OpTensorSpec weight_tensor(int idx, OpSlotOptions option = OpSlotOptions::NECESSARY) {
  return {TensorRole::WEIGHT, option, idx};
}

// OpTaskBinding::OpTaskBinding() {
//   this->serializer.reserve_bytes(sizeof(TaskArgumentFormat));
// }

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
    if (slot_option != OpSlotOptions::UNTRAINABLE || slot_option != OpSlotOptions::OPTIONAL_UNTRAINABLE) {
      slot_id slot = key.first;
      bwd.bind_grad(slot, spec);
    }
  }
  return bwd;
}

} // namespace FlexFlow

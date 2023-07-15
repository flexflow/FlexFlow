#include "op_task_invocation.h"
#include "task_argument_accessor.h"

namespace FlexFlow {

OpTaskSignature get_signature(task_id_t const &) {
  NOT_IMPLEMENTED();
}

OpTensorSpec::OpTensorSpec(TensorRole _role, int _idx)
    : role(_role), idx(_idx) {}

OpTensorSpec input_tensor(int idx) {
  return {TensorRole::INPUT, idx};
}

OpTensorSpec output_tensor(int idx) {
  return {TensorRole::OUTPUT, idx};
}

OpTensorSpec weight_tensor(int idx) {
  return {TensorRole::WEIGHT, idx};
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

std::unordered_map<slot_id, OpTaskBinding::ArgSpec> const &
    OpTaskBinding::get_arg_bindings() const {
  return this->arg_bindings;
}

} // namespace FlexFlow

#include "local-execution/task_invocation.h"
#include "utils/containers/contains_key.h"

namespace FlexFlow {

void TaskBinding::bind(int name, TensorGuidSpec const &tensor_guid_spec) {
  this->bind(slot_id_t{name}, tensor_guid_spec);
}

void TaskBinding::bind(slot_id_t name, TensorGuidSpec const &tensor_guid_spec) {
  this->tensor_bindings.insert(
      {SlotGradId{name, tensor_guid_spec.is_grad}, tensor_guid_spec});
}

void TaskBinding::insert_arg_spec(slot_id_t name, TaskArgSpec const &arg_spec) {
  assert(!contains_key(this->arg_bindings, name));
  this->arg_bindings.insert({name, arg_spec});
}

bool TaskBinding::operator==(TaskBinding const &other) const {
  return this->tie() == other.tie();
}

bool TaskBinding::operator!=(TaskBinding const &other) const {
  return this->tie() != other.tie();
}

std::tuple<std::unordered_map<SlotGradId, TensorGuidSpec> const &,
           std::unordered_map<slot_id_t, TaskArgSpec> const &>
    TaskBinding::tie() const {
  return std::tie(this->tensor_bindings, this->arg_bindings);
}

std::unordered_map<SlotGradId, TensorGuidSpec> const &
    TaskBinding::get_tensor_bindings() const {
  return this->tensor_bindings;
}

std::unordered_map<slot_id_t, TaskArgSpec> const &
    TaskBinding::get_arg_bindings() const {
  return this->arg_bindings;
}

bool is_invocation_valid(TaskSignature const &sig, TaskInvocation const &inv) {
  // TODO: implement signature checking
  return true;
}

} // namespace FlexFlow

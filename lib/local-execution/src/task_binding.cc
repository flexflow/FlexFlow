#include "local-execution/task_binding.h"
#include "utils/containers/contains_key.h"
#include "utils/fmt/unordered_map.h"

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

std::string format_as(TaskBinding const &x) {
  std::ostringstream oss;
  oss << "<TaskBinding";
  oss << " tensor_bindings=" << x.get_tensor_bindings();
  oss << " arg_bindings=" << x.get_arg_bindings();
  return oss.str();
}

std::ostream &operator<<(std::ostream &s, TaskBinding const &x) {
  return (s << fmt::to_string(x));
}

} // namespace FlexFlow

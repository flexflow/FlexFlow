#include "task_invocation.h"

namespace FlexFlow {

void TaskBinding::insert_arg_spec(slot_id name, ArgSpec const &arg_spec) {
  assert(!contains_key(this->arg_bindings, name));
  this->arg_bindings.insert({name, arg_spec});
}

} // namespace FlexFlow

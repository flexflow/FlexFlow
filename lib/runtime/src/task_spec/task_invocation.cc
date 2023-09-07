#include "task_invocation.h"
#include "legion.h"
#include "task_signature.h"

using namespace Legion;

namespace FlexFlow {

void TaskBinding::insert_arg_spec(slot_id name, ArgSpec const &arg_spec) {
  assert(!contains_key(this->arg_bindings, name));
  this->arg_bindings.insert({name, arg_spec});
}

TaskSignature get_signature(task_id_t task_id) {
  return TaskSignature::task_sig_map.at(task_id);
}


} // namespace FlexFlow

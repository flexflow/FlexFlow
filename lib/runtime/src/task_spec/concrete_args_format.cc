#include "concrete_args_format.h"

namespace FlexFlow {

ConcreteArgsFormat process_concrete_args(
    std::unordered_map<slot_id, ConcreteArgSpec> const &specs) {
  Legion::Serializer sez;
  TaskArgumentsFormat *reserved = static_cast<TaskArgumentsFormat *>(
      sez.reserve_bytes(sizeof(TaskArgumentsFormat)));
  stack_map<slot_id, TaskArgumentFormat, MAX_NUM_TASK_ARGUMENTS> fmts;
  for (auto const &kv : specs) {
    slot_id slot = kv.first;
    ConcreteArgSpec arg = kv.second;

    size_t before = sez.get_used_bytes();
    arg.serialize(sez);
    size_t after = sez.get_used_bytes();

    fmts.insert(slot, {arg.get_type_tag().get_type_idx(), before, after});
  }
  return {sez, reserved, fmts};
}

ConcreteArgsFormat process_concrete_args(TensorlessTaskBinding const &binding) {
  return process_concrete_args(get_args_of_type<ConcreteArgSpec>(binding));
}

} // namespace FlexFlow

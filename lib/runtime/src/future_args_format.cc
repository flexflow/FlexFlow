#include "future_args_format.h"

namespace FlexFlow {

FutureArgsFormat process_future_args(TensorlessTaskBinding const &binding) {
  std::vector<Legion::Future> futures;
  stack_map<slot_id, FutureArgumentFormat, MAX_NUM_TASK_ARGUMENTS> fmts;

  for (auto const &kv : get_args_of_type<CheckedTypedFuture>(binding)) {
    slot_id slot = kv.first;
    CheckedTypedFuture fut = kv.second;

    futures.push_back(fut.get_unsafe());
    FutureArgumentFormat fmt = {fut.get_type_idx(), futures.size() - 1};
    fmts.insert(slot, fmt);
  }
  return {futures, fmts};
};

} // namespace FlexFlow

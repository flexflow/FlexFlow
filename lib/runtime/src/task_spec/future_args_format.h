#ifndef _FLEXFLOW_RUNTIME_SRC_FUTURE_ARGS_FORMAT_H
#define _FLEXFLOW_RUNTIME_SRC_FUTURE_ARGS_FORMAT_H

#include "legion.h"
#include "runtime/task_spec/tensorless_task_invocation.h"
#include "task_argument_accessor.h"
#include "utils/stack_map.h"

namespace FlexFlow {

struct FutureArgsFormat {
  std::vector<Legion::Future> futures;
  stack_map<slot_id, FutureArgumentFormat, MAX_NUM_TASK_ARGUMENTS> fmts;
};

FutureArgsFormat process_future_args(TensorlessTaskBinding const &);
CheckedTypedFuture resolve_future_map_arg(CheckedTypedFuture const &,
                                          Legion::Domain const &);

} // namespace FlexFlow

#endif

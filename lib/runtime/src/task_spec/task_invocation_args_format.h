#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_INVOCATION_ARGS_FORMAT_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_INVOCATION_ARGS_FORMAT_H

#include "runtime/legion_backing.h"
#include "runtime/task_spec/tensorless_task_invocation.h"

namespace FlexFlow {

struct TaskInvocationArgsFormat {};

std::vector<StandardExecutableArgSpec> process_task_invocation_args(
    TensorlessTaskBinding const &, EnableProfiling, LegionBacking const &);

} // namespace FlexFlow

#endif

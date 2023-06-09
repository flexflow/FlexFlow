#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_INVOCATION_ARGS_FORMAT_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_INVOCATION_ARGS_FORMAT_H

#include "legion_backing.h"
#include "tensorless_task_invocation.h"

namespace FlexFlow {

struct TaskInvocationArgsFormat {};

std::vector<ExecutableArgSpec> process_task_invocation_args(
    TensorlessTaskBinding const &, EnableProfiling, RuntimeBacking const &);

} // namespace FlexFlow

#endif

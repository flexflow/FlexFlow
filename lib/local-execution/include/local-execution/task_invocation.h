#ifndef _FLEXFLOW_LOCAL_EXECUTION_TASK_INVOCATION_H
#define _FLEXFLOW_LOCAL_EXECUTION_TASK_INVOCATION_H

#include "local-execution/task_invocation.dtg.h"

namespace FlexFlow {

bool is_invocation_valid(TaskSignature const &sig, TaskInvocation const &inv);

}

#endif

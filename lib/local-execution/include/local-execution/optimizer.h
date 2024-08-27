#ifndef _FLEXFLOW_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_OPTIMIZER_H_
#define _FLEXFLOW_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_OPTIMIZER_H_

#include "local-execution/task_impl_function.dtg.h"
#include "local-execution/task_invocation.h"
#include "local-execution/task_signature.h"
#include "pcg/optimizers/sgd_optimizer_attrs.dtg.h"
#include "pcg/optimizers/adam_optimizer_attrs.dtg.h"

namespace FlexFlow {

TaskSignature get_sgd_update_signature();
TaskInvocation sgd_update(SGDOptimizerAttrs const &);
TaskImplFunction get_sgd_update_task_impl();

TaskSignature get_adam_update_signature();
TaskInvocation adam_update(SGDOptimizerAttrs const &);
TaskImplFunction get_adam_update_task_impl();

} // namespace FlexFlow

#endif

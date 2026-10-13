#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_SERIALIZABLE_NCCL_TASK_ARGS_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_SERIALIZABLE_NCCL_TASK_ARGS_H

#include "realm-execution/tasks/impl/nccl_task_args.dtg.h"
#include "realm-execution/tasks/impl/serializable_nccl_task_args.dtg.h"

namespace FlexFlow {

SerializableNcclTaskArgs
    nccl_task_args_to_serializable(NCCLTaskArgs const &);

NCCLTaskArgs
    nccl_task_args_from_serializable(SerializableNcclTaskArgs const &);

} // namespace FlexFlow

#endif

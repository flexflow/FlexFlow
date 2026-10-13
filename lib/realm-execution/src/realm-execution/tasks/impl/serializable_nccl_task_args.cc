#include "realm-execution/tasks/impl/serializable_nccl_task_args.h"

namespace FlexFlow {

SerializableNcclTaskArgs
    nccl_task_args_to_serializable(NcclTaskArgs const &args) {
  return SerializableNcclTaskArgs{
      args.message,
  };
}

NcclTaskArgs
    nccl_task_args_from_serializable(SerializableNcclTaskArgs const &args) {
  return NcclTaskArgs{
      args.message,
  };
}

} // namespace FlexFlow

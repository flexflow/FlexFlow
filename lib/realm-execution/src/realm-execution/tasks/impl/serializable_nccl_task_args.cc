#include "realm-execution/tasks/impl/serializable_nccl_task_args.h"
#include "realm-execution/tasks/serializer/serializable_device_specific_ptr.h"
#include "realm-execution/tasks/serializer/serializable_tensor_instance_backing.h"
#include "task-spec/dynamic_graph/serializable_dynamic_node_invocation.h"

namespace FlexFlow {

SerializableNcclTaskArgs
nccl_task_args_to_serializable(NCCLTaskArgs const &args) {
  return SerializableNcclTaskArgs{
      /*invocation=*/
      dynamic_node_invocation_to_serializable(args.invocation),
      /*tensor_backing=*/
      tensor_instance_backing_to_serializable(args.tensor_backing),
      /*device_handle=*/
      device_specific_ptr_to_serializable(args.device_handle),
  };
}

NCCLTaskArgs
nccl_task_args_from_serializable(SerializableNcclTaskArgs const &args) {
  return NCCLTaskArgs{
      /*invocation=*/
      dynamic_node_invocation_from_serializable(args.invocation),
      /*tensor_backing=*/
      tensor_instance_backing_from_serializable(args.tensor_backing),
      /*device_handle=*/
      device_specific_ptr_from_serializable<ManagedPerDeviceFFHandle>(
          args.device_handle),
  };
}

} // namespace FlexFlow

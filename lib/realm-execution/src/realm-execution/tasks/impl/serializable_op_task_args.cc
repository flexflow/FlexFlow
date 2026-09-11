#include "realm-execution/tasks/impl/serializable_op_task_args.h"
#include "realm-execution/tasks/serializer/serializable_device_specific_ptr.h"
#include "realm-execution/tasks/serializer/serializable_tensor_instance_backing.h"
#include "task-spec/dynamic_graph/serializable_dynamic_node_invocation.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

SerializableOpTaskArgs op_task_args_to_serializable(OpTaskArgs const &args) {
  return SerializableOpTaskArgs{
      /*invocation=*/dynamic_node_invocation_to_serializable(args.invocation),
      /*tensor_backing*/
      tensor_instance_backing_to_serializable(args.tensor_backing),
      /*device_state=*/
      transform(args.device_state,
                device_specific_ptr_to_serializable<PerDeviceOpState>),
      /*device_handle=*/device_specific_ptr_to_serializable(args.device_handle),
      /*optimizer_attrs=*/args.optimizer_attrs,
  };
}

OpTaskArgs op_task_args_from_serializable(SerializableOpTaskArgs const &args) {
  return OpTaskArgs{
      /*invocation=*/dynamic_node_invocation_from_serializable(args.invocation),
      /*tensor_backing*/
      tensor_instance_backing_from_serializable(args.tensor_backing),
      /*device_state=*/
      transform(args.device_state,
                device_specific_ptr_from_serializable<PerDeviceOpState>),
      /*device_handle=*/
      device_specific_ptr_from_serializable<ManagedPerDeviceFFHandle>(
          args.device_handle),
      /*optimizer_attrs=*/args.optimizer_attrs,
  };
}

} // namespace FlexFlow

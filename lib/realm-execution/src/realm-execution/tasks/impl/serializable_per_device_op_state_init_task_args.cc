#include "realm-execution/tasks/impl/serializable_per_device_op_state_init_task_args.h"
#include "realm-execution/tasks/serializer/serializable_device_specific_ptr.h"
#include "realm-execution/tasks/serializer/serializable_realm_processor.h"
#include "realm-execution/tasks/serializer/serializable_tensor_instance_backing.h"
#include "task-spec/dynamic_graph/serializable_dynamic_node_invocation.h"

namespace FlexFlow {

SerializablePerDeviceOpStateInitTaskArgs
    per_device_op_state_init_task_args_to_serializable(
        PerDeviceOpStateInitTaskArgs const &args) {
  return SerializablePerDeviceOpStateInitTaskArgs{
      /*invocation=*/dynamic_node_invocation_to_serializable(args.invocation),
      /*tensor_backing*/
      tensor_instance_backing_to_serializable(args.tensor_backing),
      /*device_handle=*/device_specific_ptr_to_serializable(args.device_handle),
      /*optimizer_attrs=*/args.optimizer_attrs,
      /*origin_proc=*/realm_processor_to_serializable(args.origin_proc),
      /*origin_result_ptr=*/reinterpret_cast<uintptr_t>(args.origin_result_ptr),
  };
}

PerDeviceOpStateInitTaskArgs
    per_device_op_state_init_task_args_from_serializable(
        SerializablePerDeviceOpStateInitTaskArgs const &args) {
  return PerDeviceOpStateInitTaskArgs{
      /*invocation=*/dynamic_node_invocation_from_serializable(args.invocation),
      /*tensor_backing*/
      tensor_instance_backing_from_serializable(args.tensor_backing),
      /*device_handle=*/
      device_specific_ptr_from_serializable<ManagedPerDeviceFFHandle>(
          args.device_handle),
      /*optimizer_attrs=*/args.optimizer_attrs,
      /*origin_proc=*/realm_processor_from_serializable(args.origin_proc),
      /*origin_result_ptr=*/
      reinterpret_cast<DeviceSpecificPtr<PerDeviceOpState> *>(
          args.origin_result_ptr),
  };
}

} // namespace FlexFlow

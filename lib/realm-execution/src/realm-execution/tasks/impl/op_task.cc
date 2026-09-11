#include "realm-execution/tasks/impl/op_task.h"
#include "local-execution/task_execution.h"
#include "realm-execution/device_specific_managed_per_device_ff_handle.h"
#include "realm-execution/dynamic_tensor_accessor_from_instance.h"
#include "realm-execution/tasks/impl/op_task_args.dtg.h"
#include "realm-execution/tasks/impl/serializable_op_task_args.h"
#include "realm-execution/tasks/serializer/task_arg_serializer.h"
#include "realm-execution/tasks/task_id_t.h"
#include "task-spec/per_device_op_state.dtg.h"
#include "task-spec/per_device_op_state.h"
#include "task-spec/permissions.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"
#include "utils/optional.h"
#include <type_traits>

namespace FlexFlow {

void op_task_body(void const *args,
                  size_t arglen,
                  void const *userdata,
                  size_t userlen,
                  Realm::Processor proc) {
  OpTaskArgs task_args = op_task_args_from_serializable(
      deserialize_task_args<SerializableOpTaskArgs>(args, arglen));

  RealmContext ctx{proc};
  device_handle_t device_handle =
      device_handle_t_from_device_specific_managed_ff_handle(
          task_args.device_handle, ctx.get_current_global_device_id());

  // Patch the invocation to include the provided instances
  auto map_instance_to_accessor = [&](DynamicValueAttrs const &value) {
    DynamicValueAttrs result = value;
    auto const &[inst, event] = task_args.tensor_backing.backing.at(value);
    result.accessor = dynamic_tensor_accessor_from_instance(
        inst,
        event,
        assert_unwrap(value.parallel_tensor_shape),
        Permissions::RW, // FIXME: get real permissions?
        ctx.get_current_processor());
    return result;
  };
  DynamicNodeInvocation invocation = task_args.invocation;
  invocation.inputs = map_values(invocation.inputs, map_instance_to_accessor);
  invocation.outputs = map_values(invocation.outputs, map_instance_to_accessor);

  execute_dynamic_node_invocation(
      /*invocation=*/invocation,
      /*allocator=*/ctx.get_current_device_allocator(),
      /*profiling_settings=*/std::nullopt,
      /*ff_handle=*/device_handle,
      /*per_device_op_state=*/
      transform(and_then(task_args.device_state,
                         [&](DeviceSpecificPtr<PerDeviceOpState> const &d) {
                           return d.get(ctx.get_current_global_device_id());
                         }),
                [](PerDeviceOpState *ptr) { return *ptr; }),
      /*optimizer_attrs=*/task_args.optimizer_attrs,
      /*global_device_id=*/ctx.get_current_global_device_id());
}

Realm::Event spawn_op_task(
    RealmContext &ctx,
    Realm::Processor target_proc,
    DynamicNodeInvocation const &invocation,
    TensorInstanceBacking const &tensor_backing,
    std::optional<DeviceSpecificPtr<PerDeviceOpState>> const &device_state,
    DeviceSpecificPtr<ManagedPerDeviceFFHandle> const &device_handle,
    std::optional<OptimizerAttrs> const &optimizer_attrs,
    Realm::Event precondition) {

  OpTaskArgs task_args = OpTaskArgs{
      invocation,
      tensor_backing,
      device_state,
      device_handle,
      optimizer_attrs,
  };

  std::string serialized_args =
      serialize_task_args(op_task_args_to_serializable(task_args));

  return ctx.spawn_task(
      target_proc,
      assert_unwrap(get_task_id_for_op(invocation.node_attrs, optimizer_attrs)),
      serialized_args.data(),
      serialized_args.size(),
      Realm::ProfilingRequestSet{},
      precondition);
}

} // namespace FlexFlow

#include "realm-execution/tasks/impl/per_device_op_state_init_task.h"
#include "local-execution/per_device_op_state_initialization.h"
#include "realm-execution/dynamic_tensor_accessor_from_instance.h"
#include "realm-execution/tasks/impl/per_device_op_state_init_return_task.h"
#include "realm-execution/tasks/impl/per_device_op_state_init_task_args.dtg.h"
#include "realm-execution/tasks/impl/serializable_per_device_op_state_init_task_args.h"
#include "realm-execution/tasks/serializer/task_arg_serializer.h"
#include "realm-execution/tasks/task_id_t.dtg.h"
#include "realm-execution/tasks/task_id_t.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/per_device_op_state.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"
#include "utils/optional.h"
#include <optional>
#include <type_traits>

namespace FlexFlow {

void per_device_op_state_init_task_body(void const *args,
                                        size_t arglen,
                                        void const *userdata,
                                        size_t userlen,
                                        Realm::Processor proc) {
  PerDeviceOpStateInitTaskArgs task_args =
      per_device_op_state_init_task_args_from_serializable(
          deserialize_task_args<SerializablePerDeviceOpStateInitTaskArgs>(
              args, arglen));

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

  DynamicNodeInvocation result_invocation =
      initialize_node(invocation,
                      ctx.get_current_device_allocator(),
                      device_handle,
                      task_args.optimizer_attrs,
                      ctx.get_current_global_device_id());
  DeviceSpecificPerDeviceOpState result_state =
      assert_unwrap(result_invocation.node_attrs.per_device_op_state);
  // Important: to make sure this doesn't get deallocated, we intentionally leak
  // the allocation here
  PerDeviceOpState *result_state_ptr =
      new PerDeviceOpState{get_per_device_op_state_from_device_specific(
          result_state, ctx.get_current_global_device_id())};
  DeviceSpecificPtr<PerDeviceOpState> result_device_specific{
      ctx.get_current_global_device_id(), result_state_ptr};
  spawn_per_device_op_state_init_return_task(ctx,
                                             task_args.origin_proc,
                                             result_device_specific,
                                             task_args.origin_result_ptr,
                                             Realm::Event::NO_EVENT);
}

std::optional<Realm::Event> spawn_per_device_op_state_init_task(
    RealmContext &ctx,
    Realm::Processor target_proc,
    DynamicNodeInvocation const &invocation,
    TensorInstanceBacking const &tensor_backing,
    DeviceSpecificPtr<ManagedPerDeviceFFHandle> const &device_handle,
    OptimizerAttrs const &optimizer_attrs,
    DeviceSpecificPtr<PerDeviceOpState> *result_ptr,
    Realm::Event precondition) {
  PerDeviceOpStateInitTaskArgs task_args{
      invocation,
      tensor_backing,
      device_handle,
      optimizer_attrs,
      ctx.get_current_processor(),
      result_ptr,
  };

  std::optional<task_id_t> task_id =
      and_then(and_then(invocation.node_attrs.op_attrs,
                        [](TrainingOperationAttrs const &op_attrs) {
                          return op_attrs.try_require_pcg_op();
                        }),
               get_init_task_id_for_op_attrs);
  if (task_id.has_value()) {
    std::string args = serialize_task_args(
        per_device_op_state_init_task_args_to_serializable(task_args));
    return ctx.spawn_task(target_proc,
                          assert_unwrap(task_id),
                          args.data(),
                          args.size(),
                          Realm::ProfilingRequestSet{},
                          precondition);
  }
  return std::nullopt;
}

} // namespace FlexFlow

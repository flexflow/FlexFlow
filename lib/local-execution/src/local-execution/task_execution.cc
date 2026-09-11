#include "local-execution/task_execution.h"
#include "local-execution/local_task_argument_accessor.h"
#include "local-execution/local_task_registry.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "pcg/optimizer_attrs.h"
#include "pcg/optimizer_slot_name.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_slot.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/dynamic_graph/training_operation_attrs.dtg.h"
#include "task-spec/task_argument_accessor/task_tensor_parameter.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/map_keys_and_values.h"
#include "utils/exception.h"
#include "utils/optional.h"
#include "utils/overload.h"
#include <optional>

namespace FlexFlow {

TaskTensorParameter make_task_tensor_parameter_from_dynamic_slot(
    DynamicTensorSlot const &slot,
    std::optional<OptimizerAttrs> const &optimizer_attrs) {
  return assert_unwrap(slot.slot_tensor_role)
      .visit<TaskTensorParameter>(overload{
          [&](FwbTensorType const &fwb_tensor) {
            switch (fwb_tensor) {
              case FwbTensorType::FORWARD:
                return make_task_tensor_parameter_fwd(slot.slot_name);
              case FwbTensorType::GRADIENT:
                return make_task_tensor_parameter_grad(slot.slot_name);
              default:
                PANIC("Unhandled FwbTensorType", fmt::to_string(fwb_tensor));
            }
          },
          [&](DynamicOptimizerTensorRole const &optimizer_tensor) {
            return make_task_tensor_parameter_opt(
                slot.slot_name, optimizer_tensor.optimizer_slot_name);
          },
          [&](DynamicLossTensorRole const &loss_tensor) {
            return make_task_tensor_parameter_loss();
          },
      });
}

TaskArgumentAccessor make_task_argument_accessor_for_invocation(
    DynamicNodeInvocation const &invocation,
    Allocator &allocator,
    std::optional<ProfilingSettings> const &profiling_settings,
    device_handle_t const &ff_handle,
    std::optional<PerDeviceOpState> const &per_device_op_state,
    std::optional<OptimizerAttrs> const &optimizer_attrs,
    global_device_id_t device_idx) {
  auto make_param = [&](DynamicTensorSlot const &slot) {
    return make_task_tensor_parameter_from_dynamic_slot(slot, optimizer_attrs);
  };
  auto get_accessor = [](DynamicValueAttrs const &value) {
    return assert_unwrap(value.accessor);
  };
  std::map<TaskTensorParameter, DynamicTensorAccessor> tensor_slots_backing =
      binary_merge_disjoint_maps(
          map_keys_and_values(invocation.inputs, make_param, get_accessor),
          map_keys_and_values(invocation.outputs, make_param, get_accessor));

  return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(
      /*allocator=*/allocator,
      /*tensor_slots_backing=*/tensor_slots_backing,
      /*profiling_settings=*/profiling_settings,
      /*ff_handle=*/ff_handle,
      /*op_attrs=*/
      and_then(invocation.node_attrs.op_attrs,
               [](TrainingOperationAttrs const &op_attrs) {
                 return op_attrs.try_require_pcg_op();
               }),
      /*loss_attrs=*/
      and_then(invocation.node_attrs.op_attrs,
               [](TrainingOperationAttrs const &op_attrs) {
                 return op_attrs.try_require_loss();
               }),
      /*per_device_op_state=*/per_device_op_state,
      /*optimizer_attrs=*/optimizer_attrs,
      /*device_idx=*/device_idx);
}

std::optional<milliseconds_t> execute_dynamic_node_invocation(
    DynamicNodeInvocation const &invocation,
    Allocator &allocator,
    std::optional<ProfilingSettings> const &profiling_settings,
    device_handle_t const &ff_handle,
    std::optional<PerDeviceOpState> const &per_device_op_state,
    std::optional<OptimizerAttrs> const &optimizer_attrs,
    global_device_id_t device_idx) {
  TaskArgumentAccessor arg_accessor =
      make_task_argument_accessor_for_invocation(
          /*invocation=*/invocation,
          /*allocator=*/allocator,
          /*profiling_settings=*/profiling_settings,
          /*ff_handle=*/ff_handle,
          /*per_device_op_state=*/per_device_op_state,
          /*optimizer_attrs=*/optimizer_attrs,
          /*device_idx=*/device_idx);

  DynamicTaskType task_type = assert_unwrap(invocation.node_attrs.task_type);
  std::optional<milliseconds_t> result;
  switch (task_type) {
    case DynamicTaskType::FWD: {
      ComputationGraphOpAttrs op_attrs =
          assert_unwrap(compgraph_op_attrs_from_pcg_op_attrs(
              assert_unwrap(invocation.node_attrs.op_attrs).require_pcg_op()));
      result = call_fwd_task_impl(op_attrs, arg_accessor);
    } break;
    case DynamicTaskType::BWD: {
      ComputationGraphOpAttrs op_attrs =
          assert_unwrap(compgraph_op_attrs_from_pcg_op_attrs(
              assert_unwrap(invocation.node_attrs.op_attrs).require_pcg_op()));
      result = call_bwd_task_impl(op_attrs, arg_accessor);
    } break;
    case DynamicTaskType::UPD:
      call_update_task_impl(assert_unwrap(optimizer_attrs), arg_accessor);
      break;
    case DynamicTaskType::LOSS:
      call_loss_task_impl(arg_accessor);
      break;
    default:
      PANIC("Unhandled DynamicTaskType", task_type);
  }
  return result;
}

} // namespace FlexFlow

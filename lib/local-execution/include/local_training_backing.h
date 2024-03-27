#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_BACKING_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_BACKING_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "kernels/linear_kernels.h"
#include "op-attrs/operator_attrs.h"
#include "op_task_signature.h"
#include "op_task_invocation.h"
#include "pcg/computation_graph.h"
#include "arg_backing.h"
#include "task_argument_accessor.h"
#include <functional>
#include <unordered_map>
#include <vector>
#include "device_specific.h"

namespace FlexFlow {

struct OperatorSlotBackingId {
  operator_guid_t op;
  slot_id slot;
};
FF_VISITABLE_STRUCT(OperatorSlotBackingId,
                    op,
                    slot);

// TODO: define device state variant in another file
// using DeviceStates = std::variant<LinearPerDeviceState>;

// using TaskImplFunction = std::variant<
//     std::function<DeviceSpecific<DeviceStates>(TaskArgumentAccessor const &)>,
//     std::function<std::optional<float>(TaskArgumentAccessor const &)>>;

struct TaskSignatureImpl {
  TaskImplFunction impl_function;
  OpTaskSignature task_signature;
};

struct TaskRegistry {
  TaskRegistry(std::unordered_map<OperatorSlotBackingId, GenericTensorAccessorW &>);
  void register_task(task_id_t, operator_guid_t);
  void register_args(operator_guid_t, OpArgBacking);
  bool is_tensor_allocated(OperatorSlotBackingId src_op_slot,
                           OperatorSlotBackingId dst_op_slot);
  GenericTensorAccessorW & get_tensor_backing(OperatorSlotBackingId op_slot_id);
  OpArgBacking get_arg_backing(operator_guid_t op_slot_id);

  OpTaskSignature get_init_signature(operator_guid_t);
  OpTaskSignature get_fwd_signature(operator_guid_t);
  OpTaskSignature get_bwd_signature(operator_guid_t);

  std::unordered_map<operator_guid_t, task_id_t> init_task_ids;
  std::unordered_map<operator_guid_t, task_id_t> forward_task_ids;
  std::unordered_map<operator_guid_t, task_id_t> backward_task_ids;
  std::unordered_map<operator_guid_t, OpArgBacking> arg_mapping;
  std::unordered_map<task_id_t, TaskSignatureImpl &> task_mapping;
  std::unordered_map<OperatorSlotBackingId, GenericTensorAccessorW &>
      tensor_mapping;
};

struct LocalTrainingBacking {
  LocalTrainingBacking(ComputationGraph,
                       Allocator,
                       std::unordered_map<OperatorSlotBackingId,
                                          GenericTensorAccessorW &>,
                       ArgBackingMapping);
  ~LocalTrainingBacking() = default;

  void execute_init();
  void execute_forward();
  void execute_backward();
  void execute_update();

  DeviceSpecific<DeviceStates> call_init_task_impl(task_id_t, TaskArgumentAccessor);
  void call_task_impl(task_id_t, TaskArgumentAccessor);

  TaskArgumentAccessor get_task_arg_accessor(OpTaskInvocation);

private:
  Allocator allocator;
  ComputationGraph computation_graph;
  TaskRegistry task_registry;
  ArgBackingMapping arg_backing_mapping;
};

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_BACKING_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_BACKING_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "kernels/linear_kernels.h"
#include "op-attrs/operator_attrs.h"
#include "op_task_signature.h"
#include "pcg/computation_graph.h"
#include "task_argument_accessor.h"
#include <functional>
#include <unordered_map>
#include <vector>

namespace FlexFlow {

struct OperatorSlotBackingId {
  operator_guid_t op;
  slot_id slot;
};

// TODO: define device state variant in another file
using DeviceStates = std::variant<LinearPerDeviceState>;

using TaskImplFunction = std::variant<
    std::function<DeviceSpecific<DeviceStates>(TaskArgumentAccessor const &)>,
    std::function<optional<float>(TaskArgumentAccessor const &)>>;

struct TaskSignatureImpl {
  TaskImplFunction impl_function;
  OpTaskSignature task_signature;
};

struct TaskRegistry {
  TaskRegistry ();
  void register_task (task_id_t);
  bool is_tensor_allocated(OperatorSlotBackingId src_op_slot, OperatorSlotBackingId dst_op_slot);
  void get_tensor_backing(OperatorSlotBackingId op_slot_id);

  std::unordered_map<task_id_t, TaskSignatureImpl> task_mapping;
  std::unordered_map<OperatorSlotBackingId, GenericTensorAccessorW> tensor_mapping;
  std::unordered_map<OperatorSlotBackingId, std::type_index> arg_mapping;
};

struct LocalTrainingBacking {
  LocalTrainingBacking(ComputationGraph const &,
                       Allocator const &,
                       std::unordered_map<OperatorSlotBackingId,
                                          GenericTensorAccessorW> const &);
  ~LocalTrainingBacking() = default;

  void execute_init();
  void execute_forward();
  void execute_backward();
  void execute_update();

  void call_task_impl(task_id_t, TaskArgumentAccessor);

  TaskArgumentAccessor get_task_arg_accessor(OpTaskInvocation);

private:
  Allocator const & allocator;
  ComputationGraph const & computation_graph;
  TaskRegistry const & task_registry;
};

} // namespace FlexFlow

#endif

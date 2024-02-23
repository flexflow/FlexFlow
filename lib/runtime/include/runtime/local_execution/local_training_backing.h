#ifndef _FLEXFLOW_RUNTIME_SRC_LOCAL_BACKING_H
#define _FLEXFLOW_RUNTIME_SRC_LOCAL_BACKING_H

#include "pcg/computation_graph.h"
#include "kernels/allocation.h"
#include "kernels/accessor.h"
#include "op-attrs/operator_attrs.h"
#include "op_task_signature.h"
#include "task_argument_accessor.h"
#include <vector>
#include <unordered_map>
#include <functional>

namespace FlexFlow {

struct OperatorSlotBackingId {
  operator_guid_t op;
  slot_id_t slot;
};

// TODO: define device state variant

template <typename DeviceState>
using TaskImplFunction = variant<std::function<DeviceSpecific<DeviceState>(TaskArgumentAccessor const &)>,
                                 std::function<optional<float>(TaskArgumentAccessor const &)>>;

struct LocalTrainingBacking {
  LocalTrainingBacking(ComputationGraph, Allocator, std::unordered_map<OperatorSlotBackingId, GenericTensorAccessorW>);
  ~LocalTrainingBacking() = default;

  GenericTensorAccessorR execute_forward();
  void execute_backward();
  void execute_update();

  LocalTaskArgumentAccessor get_fwd_accessor(OpTaskInvocation);
  LocalTaskArgumentAccessor get_bwd_accessor(OpTaskInvocation);

private:
  ComputationGraph computation_graph;
  Allocator allocator;
  std::vector<Node> topologically_ordered_graph;

  // memory
  std::unordered_map<OperatorSlotBackingId, GenericTensorAccessorW> op_slot_tensor_mapping;

  // hold mappings
  std::unordered_map<task_id_t, void*> task_id_impl_mapping;
  // TODO: add init task mapping
  std::unordered_map<task_id_t, TaskImplFunction> task_id_impl_mapping;
  std::unordered_map<task_id_t, OpTaskSignature> task_id_signature_mapping;
};

} // namespace FlexFlow

#endif
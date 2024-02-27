#ifndef _FLEXFLOW_RUNTIME_SRC_LOCAL_BACKING_H
#define _FLEXFLOW_RUNTIME_SRC_LOCAL_BACKING_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
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

// TODO: define device state variant

template <typename DeviceState>
using TaskImplFunction = variant<
    std::function<DeviceSpecific<DeviceState>(TaskArgumentAccessor const &)>,
    std::function<optional<float>(TaskArgumentAccessor const &)>>;

template <typename DeviceState>
struct TaskSignatureImpl {
  TaskImplFunction<DeviceState> impl_function;
  OpTaskSignature task_signature;
}

struct LocalTrainingBacking {
  LocalTrainingBacking(ComputationGraph const &,
                       Allocator const &,
                       std::unordered_map<OperatorSlotBackingId,
                                          GenericTensorAccessorW> const &);
  ~LocalTrainingBacking() = default;

  GenericTensorAccessorR execute_forward();
  void execute_backward();
  void execute_update();

  TaskArgumentAccessor get_task_argument_accessor(OpTaskInvocation);

private:
  Allocator allocator;
  std::vector<Node> topologically_ordered_graph;

  // memory
  std::unordered_map<OperatorSlotBackingId, GenericTensorAccessorW>
      op_slot_tensor_mapping;

  // TODO: add init task mapping
  template <typename DeviceState>
  std::unordered_map<task_id_t, TaskSignatureImpl<DeviceState>> task_mapping;
};

} // namespace FlexFlow

#endif

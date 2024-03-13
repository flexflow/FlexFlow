#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_ARGUMENT_ACCESSOR_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "kernels/linear_kernels.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "task_argument_accessor.h"
#include "op_task_signature.h"
#include "arg_ref.h"
#include "device_specific.h"
#include "concrete_arg.h"
#include "config.h"
#include <unordered_map>
#include <variant>

namespace FlexFlow {

using SlotGradId = std::pair<slot_id, IsGrad>;

// TODO: define device state variant in another file
using DeviceStates = std::variant<LinearPerDeviceState>;

using OpArgRefType = std::variant<ParallelTensorShape, DeviceSpecific<DeviceStates>>;
using RuntimeArgRefType = std::variant<ProfilingSettings,
                                  DeviceSpecific<PerDeviceFFHandle>,
                                  FFIterationConfig>;

using ArgRefBacking = std::variant<OpArgRefType, RuntimeArgRefType, ConcreteArgSpec>;

struct LocalTaskArgumentAccessor : public ITaskArgumentAccessor {

  LocalTaskArgumentAccessor() = default;

  template <typename T>
  T const &get_argument(slot_id) const;

  template <typename OpArgRefType>
  OpArgRefType const &get_argument(slot_id) const;

  template <typename RuntimeArgRefType>
  RuntimeArgRefType const &get_argument(slot_id) const;

  PrivilegeType get_tensor(slot_id slot, bool is_grad) const override;

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV> get_tensor_grad(slot_id slot) const;

  Allocator get_allocator();

  void insert_tensor(SlotGradId tensor_id,
                     GenericTensorAccessorW tensor_backing) {
    this->tensor_backing_map.insert({tensor_id, tensor_backing});
  }

private:
  Allocator allocator;
  std::unordered_map<SlotGradId, GenericTensorAccessorW> tensor_backing_map;
  std::unordered_map<slot_id, ArgRefBacking> argument_map;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalTaskArgumentAccessor);

} // namespace FlexFlow

#endif

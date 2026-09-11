#include "local-execution/local_task_argument_accessor.h"
#include "kernels/accessor.h"
#include "utils/exception.h"
#include "utils/optional.h"
#include "utils/overload.h"

namespace FlexFlow {

LocalTaskArgumentAccessor::LocalTaskArgumentAccessor(
    Allocator const &allocator,
    std::map<TaskTensorParameter, DynamicTensorAccessor> const
        &tensor_slots_backing,
    std::optional<ProfilingSettings> const &profiling_settings,
    device_handle_t const &ff_handle,
    std::optional<PCGOperatorAttrs> const &op_attrs,
    std::optional<LossAttrs> const &loss_attrs,
    std::optional<PerDeviceOpState> const &per_device_op_state,
    std::optional<OptimizerAttrs> const &optimizer_attrs,
    global_device_id_t device_idx)
    : allocator(allocator), tensor_slots_backing(tensor_slots_backing),
      profiling_settings(profiling_settings), ff_handle(ff_handle),
      op_attrs(op_attrs), loss_attrs(loss_attrs),
      per_device_op_state(per_device_op_state),
      optimizer_attrs(optimizer_attrs), device_idx(device_idx) {}

TensorShape
    LocalTaskArgumentAccessor::get_tensor_shape(TensorSlotName slot) const {

  for (auto const &[backing_slot, accessor] : this->tensor_slots_backing) {
    bool match = backing_slot.visit<bool>(overload{
        [&](TaskForwardTensorParameter const &param) {
          return param.name == slot;
        },
        [&](TaskGradientTensorParameter const &param) {
          return param.name == slot;
        },
        [&](TaskOptimizerTensorParameter const &param) {
          return param.name == slot;
        },
        [&](TaskLossTensorParameter const &param) { return false; },
    });

    if (match) {
      if (accessor.has<GenericTensorAccessorR>()) {
        return accessor.get<GenericTensorAccessorR>().shape;
      } else {
        return accessor.get<GenericTensorAccessorW>().shape;
      }
    }
  }

  PANIC("Unable to find TensorSlotName in tensor_slots_backing",
        fmt::to_string(slot));
}

GenericTensorAccessor
    LocalTaskArgumentAccessor::get_tensor(TaskTensorParameter slot,
                                          Permissions priv) const {
  DynamicTensorAccessor tensor_backing = this->tensor_slots_backing.at(slot);
  if (priv == Permissions::RO) {
    if (tensor_backing.is_read()) {
      return tensor_backing.require_read();
    } else {
      GenericTensorAccessorR readonly_tensor_backing =
          read_only_accessor_from_write_accessor(
              tensor_backing.require_write());
      return readonly_tensor_backing;
    }
  } else if (priv == Permissions::RW || priv == Permissions::WO) {
    return tensor_backing.require_write();
  } else {
    PANIC(fmt::format("Unhandled privilege mode {}", priv));
  }
}

std::optional<ProfilingSettings>
    LocalTaskArgumentAccessor::get_profiling_settings() const {
  return this->profiling_settings;
}

device_handle_t LocalTaskArgumentAccessor::get_ff_handle() const {
  return this->ff_handle;
}

DeviceType LocalTaskArgumentAccessor::get_kernel_device_type() const {
  return this->device_idx.device_type;
}

PCGOperatorAttrs LocalTaskArgumentAccessor::get_op_attrs() const {
  return assert_unwrap(this->op_attrs);
}

LossAttrs LocalTaskArgumentAccessor::get_loss_attrs() const {
  return assert_unwrap(this->loss_attrs);
}

PerDeviceOpState LocalTaskArgumentAccessor::get_per_device_op_state() const {
  return assert_unwrap(this->per_device_op_state);
}

OptimizerAttrs LocalTaskArgumentAccessor::get_optimizer_attrs() const {
  return assert_unwrap(this->optimizer_attrs);
}

Allocator LocalTaskArgumentAccessor::get_allocator() const {
  return this->allocator;
}

global_device_id_t LocalTaskArgumentAccessor::get_device_idx() const {
  return this->device_idx;
}

} // namespace FlexFlow

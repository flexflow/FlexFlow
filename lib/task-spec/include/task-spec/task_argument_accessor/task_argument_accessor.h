#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TASK_ARGUMENT_ACCESSOR_TASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TASK_ARGUMENT_ACCESSOR_TASK_ARGUMENT_ACCESSOR_H

#include "kernels/device_handle_t.dtg.h"
#include "kernels/profiling_settings.dtg.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "pcg/optimizer_slot_name.dtg.h"
#include "task-spec/device_specific.h"
#include "task-spec/per_device_op_state.dtg.h"
#include "task-spec/task_argument_accessor/itask_argument_accessor.h"
#include "task-spec/task_argument_accessor/task_tensor_parameter.h"

namespace FlexFlow {

struct TaskArgumentAccessor {
  std::optional<ProfilingSettings> get_profiling_settings() const;
  device_handle_t get_ff_handle() const;
  DeviceType get_kernel_device_type() const;
  PCGOperatorAttrs get_op_attrs() const;
  LossAttrs get_loss_attrs() const;
  PerDeviceOpState get_per_device_op_state() const;
  OptimizerAttrs get_optimizer_attrs() const;

  TensorShape get_tensor_shape(TensorSlotName slot) const {
    return this->ptr->get_tensor_shape(slot);
  }

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV> get_tensor(TensorSlotName slot) const {
    return std::get<privilege_mode_to_accessor<PRIV>>(
        this->ptr->get_tensor(make_task_tensor_parameter_fwd(slot), PRIV));
  }

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV> get_tensor_grad(TensorSlotName slot) const {
    return std::get<privilege_mode_to_accessor<PRIV>>(
        this->ptr->get_tensor(make_task_tensor_parameter_grad(slot), PRIV));
  }

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV>
      get_optimizer_tensor(TensorSlotName slot,
                           OptimizerSlotName opt_slot) const {
    return std::get<privilege_mode_to_accessor<PRIV>>(this->ptr->get_tensor(
        make_task_tensor_parameter_opt(slot, opt_slot), PRIV));
  }

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV> get_loss_tensor() const {
    return std::get<privilege_mode_to_accessor<PRIV>>(
        this->ptr->get_tensor(make_task_tensor_parameter_loss(), PRIV));
  }

  Allocator get_allocator() const {
    return this->ptr->get_allocator();
  }

  global_device_id_t get_device_idx() const {
    return this->ptr->get_device_idx();
  }

  template <typename T>
  DeviceSpecific<T> make_device_specific(T const &t) const {
    return DeviceSpecific<T>::create(this->get_device_idx(), t);
  }

  template <typename T, typename... Args>
  static
      typename std::enable_if<std::is_base_of<ITaskArgumentAccessor, T>::value,
                              TaskArgumentAccessor>::type
      create(Args &&...args) {
    return TaskArgumentAccessor(
        std::make_shared<T>(std::forward<Args>(args)...));
  }

private:
  TaskArgumentAccessor(std::shared_ptr<ITaskArgumentAccessor const> ptr)
      : ptr(ptr) {}
  std::shared_ptr<ITaskArgumentAccessor const> ptr;
};

} // namespace FlexFlow

#endif

#include "task-spec/task_argument_accessor/task_argument_accessor.h"

namespace FlexFlow {

std::optional<ProfilingSettings>
    TaskArgumentAccessor::get_profiling_settings() const {
  return this->ptr->get_profiling_settings();
}

device_handle_t TaskArgumentAccessor::get_ff_handle() const {
  return this->ptr->get_ff_handle();
}
DeviceType TaskArgumentAccessor::get_kernel_device_type() const {
  return this->ptr->get_kernel_device_type();
}

PCGOperatorAttrs TaskArgumentAccessor::get_op_attrs() const {
  return this->ptr->get_op_attrs();
}

LossAttrs TaskArgumentAccessor::get_loss_attrs() const {
  return this->ptr->get_loss_attrs();
}

PerDeviceOpState TaskArgumentAccessor::get_per_device_op_state() const {
  return this->ptr->get_per_device_op_state();
}

OptimizerAttrs TaskArgumentAccessor::get_optimizer_attrs() const {
  return this->ptr->get_optimizer_attrs();
}

} // namespace FlexFlow

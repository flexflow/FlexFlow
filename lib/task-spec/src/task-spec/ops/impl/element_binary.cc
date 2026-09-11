#include "task-spec/ops/impl/element_binary.h"
#include "kernels/element_binary_kernels.h"
#include "task-spec/profiling.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  ElementBinaryAttrs attrs = acc.get_op_attrs().require_element_binary();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  TensorShape input_lhs_shape = acc.get_tensor_shape(TensorSlotName::LHS_INPUT);
  TensorShape input_rhs_shape = acc.get_tensor_shape(TensorSlotName::RHS_INPUT);
  TensorShape output_shape = acc.get_tensor_shape(TensorSlotName::OUTPUT);

  std::optional<ElementBinaryPerDeviceState> per_device_state =
      element_binary_init_kernel(kernel_device_type,
                                 attrs,
                                 input_lhs_shape,
                                 input_rhs_shape,
                                 output_shape);

  return DeviceSpecificPerDeviceOpState{
      acc.make_device_specific(per_device_state),
  };
}

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  std::optional<ElementBinaryPerDeviceState> per_device_state =
      acc.get_per_device_op_state().require_element_binary();
  ElementBinaryAttrs attrs = acc.get_op_attrs().require_element_binary();
  device_handle_t handle = acc.get_ff_handle();

  GenericTensorAccessorR input_lhs =
      acc.get_tensor<Permissions::RO>(TensorSlotName::LHS_INPUT);
  GenericTensorAccessorR input_rhs =
      acc.get_tensor<Permissions::RO>(TensorSlotName::RHS_INPUT);
  GenericTensorAccessorW output =
      acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  return profile(element_binary_forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[ElementBinary] forward_time = {:.2lf}ms\n",
                 handle,
                 per_device_state,
                 attrs,
                 input_lhs,
                 input_rhs,
                 output);
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  std::optional<ElementBinaryPerDeviceState> per_device_state =
      acc.get_per_device_op_state().require_element_binary();
  ElementBinaryAttrs attrs = acc.get_op_attrs().require_element_binary();
  device_handle_t handle = acc.get_ff_handle();

  GenericTensorAccessorR input_lhs =
      acc.get_tensor<Permissions::RO>(TensorSlotName::LHS_INPUT);
  GenericTensorAccessorR input_rhs =
      acc.get_tensor<Permissions::RO>(TensorSlotName::RHS_INPUT);
  GenericTensorAccessorR output =
      acc.get_tensor<Permissions::RO>(TensorSlotName::OUTPUT);

  GenericTensorAccessorR output_grad =
      acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);
  GenericTensorAccessorW input_lhs_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::LHS_INPUT);
  GenericTensorAccessorW input_rhs_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::RHS_INPUT);

  return profile(element_binary_backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[ElementBinary] backward_time = {:.2lf}ms\n",
                 handle,
                 per_device_state,
                 attrs,
                 output,
                 output_grad,
                 input_lhs,
                 input_lhs_grad,
                 input_rhs,
                 input_rhs_grad);
}

TaskImplFunction get_element_binary_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}

TaskImplFunction get_element_binary_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_element_binary_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

}; // namespace FlexFlow

#include "task-spec/ops/impl/conv_2d.h"
#include "kernels/conv_2d_kernels.h"
#include "task-spec/profiling.h"

namespace FlexFlow {

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  device_handle_t handle = acc.get_ff_handle();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  Conv2DAttrs attrs = acc.get_op_attrs().require_conv2d();

  TensorShape input_shape = acc.get_tensor_shape(TensorSlotName::INPUT);
  TensorShape output_shape = acc.get_tensor_shape(TensorSlotName::OUTPUT);

  std::optional<Conv2DPerDeviceState> per_device_state = conv_2d_init_kernel(
      /*device_type=*/kernel_device_type,
      /*handle=*/handle,
      /*attrs=*/attrs,
      /*input_shape=*/input_shape,
      /*output_shape=*/output_shape);

  return DeviceSpecificPerDeviceOpState{
      acc.make_device_specific(per_device_state),
  };
}

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  device_handle_t handle = acc.get_ff_handle();
  std::optional<Conv2DPerDeviceState> per_device_state =
      acc.get_per_device_op_state().require_conv2d();
  Conv2DAttrs attrs = acc.get_op_attrs().require_conv2d();

  GenericTensorAccessorR input =
      acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  GenericTensorAccessorR filter =
      acc.get_tensor<Permissions::RO>(TensorSlotName::FILTER);
  std::optional<GenericTensorAccessorR> bias =
      attrs.use_bias ? std::optional<GenericTensorAccessorR>{acc.get_tensor<
                           Permissions::RO>(TensorSlotName::BIAS)}
                     : std::nullopt;
  GenericTensorAccessorW output =
      acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  return profile(conv_2d_forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Conv2D] forward_time = {:.2lf}ms\n",
                 handle,
                 per_device_state,
                 attrs,
                 input,
                 filter,
                 bias,
                 output);
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  device_handle_t handle = acc.get_ff_handle();
  std::optional<Conv2DPerDeviceState> per_device_state =
      acc.get_per_device_op_state().require_conv2d();
  Conv2DAttrs attrs = acc.get_op_attrs().require_conv2d();

  GenericTensorAccessorR input =
      acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  GenericTensorAccessorW input_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::INPUT);
  GenericTensorAccessorR output =
      acc.get_tensor<Permissions::RO>(TensorSlotName::OUTPUT);
  GenericTensorAccessorR output_grad =
      acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);
  GenericTensorAccessorR filter =
      acc.get_tensor<Permissions::RO>(TensorSlotName::FILTER);
  GenericTensorAccessorW filter_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::FILTER);
  std::optional<GenericTensorAccessorW> bias_grad =
      attrs.use_bias
          ? std::optional<GenericTensorAccessorW>{acc.get_tensor_grad<
                Permissions::RW>(TensorSlotName::BIAS)}
          : std::nullopt;

  return profile(conv_2d_backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Conv2D] backward_time = {:.2lf}ms\n",
                 handle,
                 per_device_state,
                 attrs,
                 output,
                 output_grad,
                 input,
                 input_grad,
                 filter,
                 filter_grad,
                 bias_grad);
}

TaskImplFunction get_conv_2d_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}

TaskImplFunction get_conv_2d_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_conv_2d_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

} // namespace FlexFlow

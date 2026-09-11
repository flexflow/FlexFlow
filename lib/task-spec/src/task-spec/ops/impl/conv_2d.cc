#include "task-spec/ops/impl/conv_2d.h"
#include "kernels/conv_2d_kernels.h"
#include "task-spec/profiling.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Conv2D;

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {

  device_handle_t handle = acc.get_ff_handle();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  Conv2DAttrs attrs = acc.get_op_attrs().require_conv2d();
  auto input = acc.get_tensor<Permissions::WO>(TensorSlotName::INPUT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);
  auto filter = acc.get_tensor<Permissions::RO>(TensorSlotName::FILTER);
  auto filter_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::FILTER);

  std::optional<Conv2DPerDeviceState> per_device_state = init_kernel(
      /*device_type=*/kernel_device_type,
      /*handle=*/handle,
      /*activation=*/attrs.activation,
      /*kernel_h=*/attrs.kernel_h.int_from_positive_int(),
      /*kernel_w=*/attrs.kernel_w.int_from_positive_int(),
      /*groups=*/attrs.groups.int_from_positive_int(),
      /*padding_h=*/attrs.padding_h.unwrap_nonnegative(),
      /*padding_w=*/attrs.padding_w.unwrap_nonnegative(),
      /*stride_h=*/attrs.stride_h.int_from_positive_int(),
      /*stride_w=*/attrs.stride_w.int_from_positive_int(),
      /*input=*/input,
      /*output=*/output,
      /*filter_ptr=*/filter.get_float_ptr(),
      /*filter_grad_ptr=*/filter_grad.get_float_ptr());

  return DeviceSpecificPerDeviceOpState{
      acc.make_device_specific(per_device_state),
  };
}

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  Conv2DPerDeviceState per_device_state =
      acc.get_per_device_op_state().require_conv2d().value();
  Conv2DAttrs attrs = acc.get_op_attrs().require_conv2d();

  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto filter = acc.get_tensor<Permissions::RO>(TensorSlotName::FILTER);
  auto bias = acc.get_tensor<Permissions::RO>(TensorSlotName::BIAS);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Conv2d] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr(),
                 filter.get_float_ptr(),
                 bias.get_float_ptr(),
                 attrs.activation);
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  Conv2DPerDeviceState per_device_state =
      acc.get_per_device_op_state().require_conv2d().value();
  Conv2DAttrs attrs = acc.get_op_attrs().require_conv2d();

  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);
  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto filter = acc.get_tensor<Permissions::RO>(TensorSlotName::FILTER);

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::INPUT);
  auto output_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::OUTPUT);
  auto filter_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::FILTER);
  auto bias_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::BIAS);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Conv2d] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 output.get_float_ptr(),
                 output_grad.get_float_ptr(),
                 input.get_float_ptr(),
                 input_grad.get_float_ptr(),
                 filter.get_float_ptr(),
                 filter_grad.get_float_ptr(),
                 bias_grad.get_float_ptr(),
                 attrs.activation);
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

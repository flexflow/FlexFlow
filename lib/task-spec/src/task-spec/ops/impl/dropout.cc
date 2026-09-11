#include "task-spec/ops/impl/dropout.h"
#include "kernels/dropout_kernels.h"
#include "task-spec/profiling.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Dropout;

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);
  Allocator allocator = acc.get_allocator();
  device_handle_t handle = acc.get_ff_handle();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  DropoutAttrs attrs = acc.get_op_attrs().require_dropout();

  std::optional<DropoutPerDeviceState> per_device_state =
      init_kernel(kernel_device_type,
                  handle,
                  attrs.rate,
                  attrs.seed,
                  output.shape,
                  allocator);

  return DeviceSpecificPerDeviceOpState{
      acc.make_device_specific(per_device_state),
  };
}

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  DropoutPerDeviceState per_device_state =
      acc.get_per_device_op_state().require_dropout().value();
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Dropout] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr());
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {

  DropoutPerDeviceState per_device_state =
      acc.get_per_device_op_state().require_dropout().value();
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::INPUT);
  auto output_grad =
      acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Dropout] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 output_grad.get_float_ptr(),
                 input_grad.get_float_ptr());
}

TaskImplFunction get_dropout_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}

TaskImplFunction get_dropout_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_dropout_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

} // namespace FlexFlow

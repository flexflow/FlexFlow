#include "task-spec/ops/impl/pool_2d.h"
#include "kernels/pool_2d_kernels.h"
#include "op-attrs/ops/pool_2d.h"
#include "task-spec/profiling.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  Pool2DAttrs attrs = acc.get_op_attrs().require_pool2d();
  device_handle_t handle = acc.get_ff_handle();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  TensorShape input_shape = acc.get_tensor_shape(TensorSlotName::INPUT);

  std::optional<Pool2DPerDeviceState> per_device_state =
      pool2d_init_kernel(kernel_device_type, handle, attrs, input_shape);

  return DeviceSpecificPerDeviceOpState{
      acc.make_device_specific(per_device_state),
  };
}

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  Pool2DAttrs attrs = acc.get_op_attrs().require_pool2d();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  Pool2DPerDeviceState state =
      acc.get_per_device_op_state().require_pool_2d().value();

  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  return profile(pool2d_forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Pool2D] forward_time = {:.2lf}ms\n",
                 state,
                 attrs,
                 input,
                 output);
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  Pool2DAttrs attrs = acc.get_op_attrs().require_pool2d();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  Pool2DPerDeviceState state =
      acc.get_per_device_op_state().require_pool_2d().value();

  auto output = acc.get_tensor<Permissions::RO>(TensorSlotName::OUTPUT);
  auto output_grad = acc.get_tensor<Permissions::RO>(TensorSlotName::OUTPUT);
  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto input_grad = acc.get_tensor<Permissions::RW>(TensorSlotName::INPUT);

  return profile(pool2d_backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Pool2D] backward_time = {:.2lf}ms\n",
                 state,
                 attrs,
                 output,
                 output_grad,
                 input,
                 input_grad);
}

TaskImplFunction get_pool_2d_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}

TaskImplFunction get_pool_2d_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_pool_2d_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

}; // namespace FlexFlow

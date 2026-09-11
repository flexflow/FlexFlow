#include "task-spec/ops/impl/reduce.h"
#include "kernels/reduce_kernels.h"
#include "task-spec/profiling.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"
#include "utils/type_traits_core.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Reduce;

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  device_handle_t handle = acc.get_ff_handle();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  ReduceAttrs attrs = acc.get_op_attrs().require_reduce();
  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  OperatorType op_type = attrs.op_type;

  nonnegative_int reduction_size =
      get_num_elements(input.shape.dims) / get_num_elements(output.shape.dims);

  std::optional<ReducePerDeviceState> per_device_state =
      init_kernel(kernel_device_type,
                  handle,
                  op_type,
                  reduction_size.unwrap_nonnegative(),
                  input.shape,
                  output.shape);

  return DeviceSpecificPerDeviceOpState{
      acc.make_device_specific(per_device_state),
  };
}

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  ReducePerDeviceState per_device_state =
      acc.get_per_device_op_state().require_reduce().value();
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Reduce] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr());
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ReducePerDeviceState per_device_state =
      acc.get_per_device_op_state().require_reduce().value();
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  auto input_grad = acc.get_tensor_grad<Permissions::WO>(TensorSlotName::INPUT);
  auto output_grad =
      acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Reduce] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 output_grad.get_float_ptr(),
                 input_grad.get_float_ptr());
}

TaskImplFunction get_reduce_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}

TaskImplFunction get_reduce_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_reduce_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

}; // namespace FlexFlow

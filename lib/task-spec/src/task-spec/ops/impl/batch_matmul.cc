#include "task-spec/ops/impl/batch_matmul.h"
#include "kernels/batch_matmul_kernels.h"
#include "task-spec/profiling.h"

namespace FlexFlow {

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {

  ProfilingSettings profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  device_handle_t handle = acc.get_ff_handle();
  auto lhs_input = acc.get_tensor<Permissions::RO>(TensorSlotName::LHS_INPUT);
  auto rhs_input = acc.get_tensor<Permissions::RO>(TensorSlotName::RHS_INPUT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  return profile(batch_matmul_forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[BatchMatmul] forward_time = {:.2lf}ms\n",
                 handle,
                 lhs_input,
                 rhs_input,
                 output);
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  device_handle_t handle = acc.get_ff_handle();

  auto lhs_input = acc.get_tensor<Permissions::RO>(TensorSlotName::LHS_INPUT);
  auto lhs_input_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::LHS_INPUT);

  auto rhs_input = acc.get_tensor<Permissions::RO>(TensorSlotName::RHS_INPUT);
  auto rhs_input_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::RHS_INPUT);

  auto output = acc.get_tensor<Permissions::RO>(TensorSlotName::OUTPUT);
  auto output_grad =
      acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);

  return profile(batch_matmul_backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[BatchMatmul] backward_time = {:.2lf}ms\n",
                 handle,
                 output,
                 output_grad,
                 lhs_input,
                 lhs_input_grad,
                 rhs_input,
                 rhs_input_grad);
}

TaskImplFunction get_batch_matmul_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_batch_matmul_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

} // namespace FlexFlow

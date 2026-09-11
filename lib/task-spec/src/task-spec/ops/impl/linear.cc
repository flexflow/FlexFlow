#include "task-spec/ops/impl/linear.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/linear_kernels.h"
#include "op-attrs/ff_dim_t.h"
#include "task-spec/profiling.h"
#include "task-spec/task_argument_accessor/task_argument_accessor.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  LinearAttrs attrs = acc.get_op_attrs().require_linear();
  device_handle_t handle = acc.get_ff_handle();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto weight = acc.get_tensor<Permissions::RO>(TensorSlotName::WEIGHT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);
  positive_int out_dim = dim_at_idx(output.shape.dims, ff_dim_t{0_n});
  positive_int batch_size = dim_at_idx(output.shape.dims, ff_dim_t{1_n});

  std::optional<LinearPerDeviceState> per_device_state =
      linear_init_kernel(kernel_device_type,
                         handle,
                         attrs.activation,
                         attrs.regularizer,
                         attrs.use_bias,
                         input.shape.data_type,
                         weight.shape.data_type,
                         output.shape.data_type,
                         batch_size.int_from_positive_int(),
                         attrs.out_channels.int_from_positive_int());

  return DeviceSpecificPerDeviceOpState{
      acc.make_device_specific(per_device_state),
  };
}

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto weight = acc.get_tensor<Permissions::RO>(TensorSlotName::WEIGHT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  LinearAttrs attrs = acc.get_op_attrs().require_linear();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  std::optional<LinearPerDeviceState> per_device_state =
      acc.get_per_device_op_state().require_linear();

  std::optional<GenericTensorAccessorR> bias = std::nullopt;
  if (attrs.use_bias) {
    bias = acc.get_tensor<Permissions::RO>(TensorSlotName::BIAS);
  }

  auto result = profile(linear_forward_kernel,
                        profiling,
                        kernel_device_type,
                        "[Linear] forward_time = {:.2lf}ms\n",
                        per_device_state,
                        attrs,
                        input,
                        output,
                        weight,
                        bias);

  return result;
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto weight = acc.get_tensor<Permissions::RO>(TensorSlotName::WEIGHT);
  auto output = acc.get_tensor<Permissions::RO>(TensorSlotName::OUTPUT);

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::INPUT);
  auto weight_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::WEIGHT);
  auto output_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::OUTPUT);

  LinearAttrs attrs = acc.get_op_attrs().require_linear();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  std::optional<LinearPerDeviceState> per_device_state =
      acc.get_per_device_op_state().require_linear();

  std::optional<GenericTensorAccessorW> bias_grad = std::nullopt;
  if (attrs.use_bias) {
    bias_grad = acc.get_tensor<Permissions::RW>(TensorSlotName::BIAS);
  }

  auto result = profile(linear_backward_kernel,
                        profiling,
                        kernel_device_type,
                        "[Linear] backward_time = {:.2lf}ms\n",
                        per_device_state,
                        attrs,
                        output,
                        output_grad,
                        input,
                        input_grad,
                        weight,
                        weight_grad,
                        bias_grad);

  return result;
}

TaskImplFunction get_linear_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}

TaskImplFunction get_linear_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_linear_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

}; // namespace FlexFlow

#include "task-spec/ops/impl/pool_2d.h"
#include "kernels/pool_2d_kernels.h"
#include "op-attrs/ops/pool_2d.h"
#include "task-spec/profiling.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"

using namespace FlexFlow::Kernels::Pool2D;

namespace FlexFlow {

static nonnegative_int calculate_padding(nonnegative_int output_size,
                                         nonnegative_int stride,
                                         nonnegative_int kernel_size,
                                         nonnegative_int input_size) {
  int o = output_size.unwrap_nonnegative();
  int s = stride.unwrap_nonnegative();
  int k = kernel_size.unwrap_nonnegative();
  int i = kernel_size.unwrap_nonnegative();

  return nonnegative_int{
      ((o - 1) * s + k - i + 1) / 2,
  };
}

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  Pool2DAttrs attrs = acc.get_op_attrs().require_pool2d();
  device_handle_t handle = acc.get_ff_handle();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  positive_int input_w = dim_at_idx(input.shape.dims, ff_dim_t{0_n});
  positive_int input_h = dim_at_idx(input.shape.dims, ff_dim_t{1_n});
  positive_int input_c = dim_at_idx(input.shape.dims, ff_dim_t{2_n});
  positive_int input_n = dim_at_idx(input.shape.dims, ff_dim_t{3_n});
  positive_int output_w = dim_at_idx(output.shape.dims, ff_dim_t{0_n});
  positive_int output_h = dim_at_idx(output.shape.dims, ff_dim_t{1_n});
  positive_int output_c = dim_at_idx(output.shape.dims, ff_dim_t{2_n});
  positive_int output_n = dim_at_idx(output.shape.dims, ff_dim_t{3_n});

  std::optional<Pool2DPerDeviceState> per_device_state =
      init_kernel(kernel_device_type,
                  handle,
                  attrs.activation,
                  input_w.int_from_positive_int(),
                  input_h.int_from_positive_int(),
                  input_c.int_from_positive_int(),
                  input_n.int_from_positive_int(),
                  output_w.int_from_positive_int(),
                  output_h.int_from_positive_int(),
                  output_c.int_from_positive_int(),
                  output_n.int_from_positive_int(),
                  attrs.padding_h.unwrap_nonnegative(),
                  attrs.padding_w.unwrap_nonnegative(),
                  attrs.kernel_h.int_from_positive_int(),
                  attrs.kernel_w.int_from_positive_int(),
                  attrs.stride_h.int_from_positive_int(),
                  attrs.stride_w.int_from_positive_int(),
                  attrs.pool_type);

  return DeviceSpecificPerDeviceOpState{
      acc.make_device_specific(per_device_state),
  };
}

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  Pool2DPerDeviceState state =
      acc.get_per_device_op_state().require_pool_2d().value();

  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Pool2D] forward_time = {:.2lf}ms\n",
                 state,
                 input.get_float_ptr(),
                 output.get_float_ptr());
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  Pool2DPerDeviceState state =
      acc.get_per_device_op_state().require_pool_2d().value();

  auto output = acc.get_tensor<Permissions::RO>(TensorSlotName::OUTPUT);
  auto output_grad = acc.get_tensor<Permissions::RO>(TensorSlotName::OUTPUT);
  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto input_grad = acc.get_tensor<Permissions::RW>(TensorSlotName::INPUT);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Pool2D] backward_time = {:.2lf}ms\n",
                 state,
                 output.get_float_ptr(),
                 output_grad.get_float_ptr(),
                 input.get_float_ptr(),
                 input_grad.get_float_ptr());
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

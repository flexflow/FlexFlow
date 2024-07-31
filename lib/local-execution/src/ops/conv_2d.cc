#include "conv_2d.h"
#include "kernels/conv_2d_kernels.h"
#include "op-attrs/get_output_shapes.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Conv2D;

enum Slots {
  INPUT,
  OUTPUT,
  FILTER,
  BIAS,
  ATTRS,
  PROFILING,
  PER_DEVICE_STATE,
  HANDLE
};

OpTaskInvocation init(Conv2DAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));
  binding.bind(FILTER, weight_tensor(0));
  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(HANDLE, ff_handle());

  return {CONV2D_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(Conv2DAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<Conv2DPerDeviceState>());

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));
  binding.bind(FILTER, weight_tensor(0));
  binding.bind(BIAS, weight_tensor(1));

  return {CONV2D_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(Conv2DAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {CONV2D_BWD_TASK_ID, binding};
}

static DeviceSpecific<DeviceStates>
    init_task_impl(TaskArgumentAccessor const &acc) {

  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);
  auto attrs = acc.get_argument<Conv2DAttrs>(ATTRS);
  auto input = acc.get_tensor<Permissions::WO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto filter = acc.get_tensor<Permissions::RO>(FILTER);
  auto filter_grad = acc.get_tensor_grad<Permissions::RW>(FILTER);

  Conv2DPerDeviceState per_device_state =
      init_kernel(handle,
                  attrs.activation,
                  attrs.kernel_h,
                  attrs.kernel_w,
                  attrs.groups,
                  attrs.padding_h,
                  attrs.padding_w,
                  attrs.stride_h,
                  attrs.stride_w,
                  input,
                  output,
                  filter.get_float_ptr(),
                  filter_grad.get_float_ptr());
  return DeviceSpecific<DeviceStates>::create(per_device_state);
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto per_device_state =
      acc.get_argument<Conv2DPerDeviceState>(PER_DEVICE_STATE);
  auto attrs = acc.get_argument<Conv2DAttrs>(ATTRS);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto filter = acc.get_tensor<Permissions::RO>(FILTER);
  auto bias = acc.get_tensor<Permissions::RO>(BIAS);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 "[Conv2d] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr(),
                 filter.get_float_ptr(),
                 bias.get_float_ptr(),
                 attrs.activation);
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto per_device_state =
      acc.get_argument<Conv2DPerDeviceState>(PER_DEVICE_STATE);
  auto attrs = acc.get_argument<Conv2DAttrs>(ATTRS);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto filter = acc.get_tensor<Permissions::RO>(FILTER);

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RW>(OUTPUT);
  auto filter_grad = acc.get_tensor_grad<Permissions::RW>(FILTER);
  auto bias_grad = acc.get_tensor_grad<Permissions::RW>(BIAS);

  return profile(backward_kernel,
                 profiling,
                 "[Conv2d] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 input_grad.get_float_ptr(),
                 output.get_float_ptr(),
                 output_grad.get_float_ptr(),
                 filter.get_float_ptr(),
                 filter_grad.get_float_ptr(),
                 bias_grad.get_float_ptr(),
                 attrs.activation);
}

TaskImplFunction get_conv_2d_init_task_impl() {
  return init_task_impl;
}
TaskImplFunction get_conv_2d_fwd_task_impl() {
  return forward_task_impl;
}
TaskImplFunction get_conv_2d_bwd_task_impl() {
  return backward_task_impl;
}

OpTaskSignature get_conv_2d_init_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_input_slot(INPUT);
  init.add_output_slot(OUTPUT);
  init.add_weight_slot(FILTER);
  init.add_arg_slot<Conv2DAttrs>(ATTRS);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  init.add_return_value<Conv2DPerDeviceState>();

  return init;
}

OpTaskSignature get_conv_2d_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<bool>(PROFILING);
  fwd.add_unchecked_arg_slot<Conv2DPerDeviceState>(PER_DEVICE_STATE);
  fwd.add_arg_slot<Conv2DAttrs>(ATTRS);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  fwd.add_weight_slot(FILTER);
  fwd.add_weight_slot(BIAS);

  return fwd;
}

OpTaskSignature get_conv_2d_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_conv_2d_fwd_signature());

  return bwd;
}

std::vector<task_id_t> get_task_ids(Conv2DAttrs const &) {
  return {CONV2D_INIT_TASK_ID, CONV2D_FWD_TASK_ID, CONV2D_BWD_TASK_ID};
}

} // namespace FlexFlow

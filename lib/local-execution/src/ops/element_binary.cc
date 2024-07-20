#include "element_binary.h"
#include "kernels/element_binary_kernels.h"
#include "local-execution/task_signature_impl.h"
#include "op-attrs/get_output_shapes.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::ElementBinary;

enum Slots {
  LHS_INPUT,
  RHS_INPUT,
  OUTPUT,
  PROFILING,
  PER_DEVICE_STATE,
  HANDLE,
  ATTRS
};

OpTaskInvocation init(ElementBinaryAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(LHS_INPUT, input_tensor(0));
  binding.bind(RHS_INPUT, input_tensor(1));
  binding.bind(OUTPUT, output_tensor(0));
  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(HANDLE, ff_handle());

  return {ELEMENTBINARY_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(ElementBinaryAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(LHS_INPUT, input_tensor(0));
  binding.bind(RHS_INPUT, input_tensor(1));
  binding.bind(OUTPUT, output_tensor(0));
  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<ElementBinaryPerDeviceState>());
  binding.bind_arg(HANDLE, ff_handle());

  return {ELEMENTBINARY_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(ElementBinaryAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {ELEMENTBINARY_BWD_TASK_ID, b};
}

static DeviceSpecific<DeviceStates>
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto input_lhs = acc.get_tensor<Permissions::RO>(LHS_INPUT);
  auto input_rhs = acc.get_tensor<Permissions::RO>(RHS_INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);
  auto const &attrs = acc.get_argument<ElementBinaryAttrs>(ATTRS);

  ElementBinaryPerDeviceState per_device_state =
      init_kernel(handle,
                  attrs.type,
                  attrs.should_broadcast_lhs,
                  attrs.should_broadcast_rhs,
                  input_lhs.shape,
                  input_rhs.shape,
                  output.shape);
  return DeviceSpecific<DeviceStates>::create(per_device_state);
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto per_device_state =
      acc.get_argument<ElementBinaryPerDeviceState>(PER_DEVICE_STATE);
  auto const &attrs = acc.get_argument<ElementBinaryAttrs>(ATTRS);

  auto input_lhs = acc.get_tensor<Permissions::RO>(LHS_INPUT);
  auto input_rhs = acc.get_tensor<Permissions::RO>(RHS_INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);

  return profile(forward_kernel,
                 profiling,
                 "[ElementBinary] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input_lhs.get_float_ptr(),
                 input_rhs.get_float_ptr(),
                 output.get_float_ptr(),
                 attrs.type,
                 attrs.should_broadcast_lhs,
                 handle);
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_argument<ElementBinaryPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto const &attrs = acc.get_argument<ElementBinaryAttrs>(ATTRS);
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);

  auto input_lhs = acc.get_tensor<Permissions::RO>(LHS_INPUT);
  auto input_rhs = acc.get_tensor<Permissions::RO>(RHS_INPUT);

  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);
  auto input_lhs_grad = acc.get_tensor_grad<Permissions::RW>(LHS_INPUT);
  auto input_rhs_grad = acc.get_tensor_grad<Permissions::RW>(RHS_INPUT);

  return profile(backward_kernel,
                 profiling,
                 "[ElementBinary] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 output_grad.get_float_ptr(),
                 input_lhs.get_float_ptr(),
                 input_rhs.get_float_ptr(),
                 input_lhs_grad.get_float_ptr(),
                 input_rhs_grad.get_float_ptr(),
                 attrs.type,
                 attrs.should_broadcast_lhs,
                 attrs.should_broadcast_rhs,
                 handle);
}

TaskImplFunction get_element_binary_init_task_impl() {
  return TaskImplFunction{init_task_impl};
}

TaskImplFunction get_element_binary_fwd_task_impl() {
  return TaskImplFunction{forward_task_impl};
}

TaskImplFunction get_element_binary_bwd_task_impl() {
  return TaskImplFunction{backward_task_impl};
}

OpTaskSignature get_element_binary_init_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_input_slot(LHS_INPUT);
  init.add_input_slot(RHS_INPUT);
  init.add_output_slot(OUTPUT);
  init.add_arg_slot<BatchMatmulAttrs>(ATTRS);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  init.add_return_value<ElementBinaryPerDeviceState>();

  return init;
}

OpTaskSignature get_element_binary_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_unchecked_arg_slot<ElementBinaryPerDeviceState>(PER_DEVICE_STATE);
  fwd.add_arg_slot<ElementBinaryAttrs>(ATTRS);
  fwd.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  fwd.add_input_slot(LHS_INPUT);
  fwd.add_input_slot(RHS_INPUT);
  fwd.add_output_slot(OUTPUT);

  return fwd;
}

OpTaskSignature get_element_binary_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_element_binary_fwd_signature());

  return bwd;
}

std::vector<task_id_t> get_task_ids(ElementBinaryAttrs const &) {
  return {ELEMENTBINARY_INIT_TASK_ID,
          ELEMENTBINARY_FWD_TASK_ID,
          ELEMENTBINARY_BWD_TASK_ID};
}

}; // namespace FlexFlow

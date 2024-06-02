#include "element_binary.h"
#include "kernels/element_binary_kernels.h"

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

static DeviceSpecific<ElementBinaryPerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto input_lhs = acc.get_tensor<Permissions::RO>(LHS_INPUT);
  auto input_rhs = acc.get_tensor<Permissions::RO>(RHS_INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);
  auto const &attrs = acc.get_argument<ElementBinaryAttrs>(ATTRS);

  DeviceSpecific<ElementBinaryPerDeviceState> per_device_state =
      init_kernel(handle,
                  attrs.type,
                  attrs.should_broadcast_lhs,
                  attrs.should_broadcast_rhs,
                  input_lhs.shape,
                  input_rhs.shape,
                  output.shape);
  return per_device_state;
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

CostMetrics
    measure_operator_cost(SimEnvFactory const &sim,
                          ElementBinaryAttrs const &attrs,
                          InputParallelTensorDesc const &input_shape_lhs,
                          InputParallelTensorDesc const &input_shape_rhs,
                          ProfilingSettings const &settings,
                          MachineView const &mv) {
  auto env = sim.new_environment();

  ParallelTensorShape output_shape =
      get_output_shape(attrs, input_shape_lhs.shape, input_shape_rhs.shape);

  SimTaskBinding init_binding;
  init_binding.bind(LHS_INPUT, input_shape_lhs);
  init_binding.bind(RHS_INPUT, input_shape_rhs);
  init_binding.bind(OUTPUT, output_shape);
  init_binding.bind_arg(ATTRS, attrs);
  init_binding.bind_arg(HANDLE, ff_handle());

  auto init_accessor =
      env.get_init_accessor(ELEMENTBINARY_INIT_TASK_ID, init_binding);
  DeviceSpecific<ElementBinaryPerDeviceState> per_device_state =
      init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;
  fwd_binding.bind(LHS_INPUT, input_shape_lhs);
  fwd_binding.bind(RHS_INPUT, input_shape_rhs);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind_arg(HANDLE, ff_handle());

  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind_arg(PER_DEVICE_STATE, per_device_state);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor =
      env.get_fwd_accessor(ELEMENTBINARY_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor =
      env.get_bwd_accessor(ELEMENTBINARY_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
OpTaskSignature init_signature<ELEMENTBINARY_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_input_slot(LHS_INPUT);
  init.add_input_slot(RHS_INPUT);
  init.add_output_slot(OUTPUT);
  init.add_arg_slot<BatchMatmulAttrs>(ATTRS);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  init.add_return_value<ElementBinaryPerDeviceState>();

  return init; // todo:this may be wrong, because the headfile retrun void
}

template <>
void register_task<ELEMENTBINARY_INIT_TASK_ID>() {
  register_task(ELEMENTBINARY_INIT_TASK_ID,
                "ElementBinary Init",
                init_signature<ELEMENTBINARY_INIT_TASK_ID>(),
                init_task_impl);
}

template <>
OpTaskSignature fwd_signature<ELEMENTBINARY_FWD_TASK_ID>() {
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

template <>
void register_task<ELEMENTBINARY_FWD_TASK_ID>() {
  register_task(ELEMENTBINARY_FWD_TASK_ID,
                "ElementBinary Fwd",
                fwd_signature<ELEMENTBINARY_FWD_TASK_ID>(),
                forward_task_impl);
}

template <>
OpTaskSignature bwd_signature<ELEMENTBINARY_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(fwd_signature<ELEMENTBINARY_FWD_TASK_ID>());

  return bwd;
}

template <>
void register_task<ELEMENTBINARY_BWD_TASK_ID>() {
  register_task(ELEMENTBINARY_BWD_TASK_ID,
                "ElementBinary Bwd",
                bwd_signature<ELEMENTBINARY_BWD_TASK_ID>(),
                backward_task_impl);
}

}; // namespace FlexFlow

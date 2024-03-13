#include "dropout.h"
#include "kernels/dropout_kernels.h"

#include "op-attrs/get_output_shapes.h"
#include "op_task_invocation.h"
#include "task_spec/task_signature.h"
#include "utils/hash-utils.h"

namespace FlexFlow {






using namespace FlexFlow::Kernels::Dropout;

enum Slots { INPUT, OUTPUT, ATTRS, PER_DEVICE_STATE, FF_HANDLE, PROFILING };

OpTaskInvocation init(DropoutAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(FF_HANDLE, ff_handle());
  binding.bind(OUTPUT, output_tensor(0));

  return {DROPOUT_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(DropoutAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<DropoutPerDeviceState>());

  return {DROPOUT_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(DropoutAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {DROPOUT_BWD_TASK_ID, b};
}

static DeviceSpecific<DropoutPerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  Allocator allocator = acc.get_allocator();
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(FF_HANDLE);
  auto const &attrs = acc.get_argument<DropoutAttrs>(ATTRS);

  DeviceSpecific<DropoutPerDeviceState> per_device_state =
          init_kernel(handle, attrs.rate, attrs.seed, output.shape, allocator);
  return per_device_state;
}


static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_argument<DropoutPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 "[Dropout] forward_time = %.2lfms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr());
}



static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  auto const &attrs = acc.get_argument<DropoutAttrs>(ATTRS);
  auto per_device_state =
      acc.get_argument<DropoutPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 "[Dropout] backward_time = %.2lfms\n",
                 per_device_state,
                 output_grad.get_float_ptr(),
                 input_grad.get_float_ptr());
}



CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  DropoutAttrs const &attrs,
                                  InputParallelTensorDesc const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv) {
  auto env = sim.new_environment();

  ParallelTensorShape output_shape = get_output_shape(attrs, input_shape.shape);

  SimTaskBinding init_binding;
  init_binding.bind_arg(FF_HANDLE, ff_handle());
  init_binding.bind_arg(ATTRS, attrs);
  init_binding.bind(OUTPUT, output_shape);

  auto init_accessor =
      env.get_init_accessor(DROPOUT_INIT_TASK_ID, init_binding);
  DeviceSpecific<DropoutPerDeviceState> per_device_state =
      init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;
  fwd_binding.bind(INPUT, input_shape);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind_arg(PER_DEVICE_STATE, per_device_state);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor = env.get_fwd_accessor(DROPOUT_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(DROPOUT_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
OpTaskSignature init_signature<DROPOUT_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);
  init.add_arg_slot<DropoutAttrs>(ATTRS);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(FF_HANDLE);
  init.add_output_slot(OUTPUT);

  init.add_return_value<DropoutPerDeviceState>();

  return init;
}

template <>
void register_task<DROPOUT_INIT_TASK_ID>() {
  register_task(DROPOUT_INIT_TASK_ID,
                "Dropout Init",
                init_signature<DROPOUT_INIT_TASK_ID>(),
                init_task);
}

template <>
OpTaskSignature fwd_signature<DROPOUT_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_unchecked_arg_slot<DropoutPerDeviceState>(PER_DEVICE_STATE);
  fwd.add_arg_slot<ProfilingSettings>(PROFILING);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  return fwd;
}

template <>
void register_task<DROPOUT_FWD_TASK_ID>() {
  register_task(DROPOUT_FWD_TASK_ID,
                "Dropout Fwd",
                fwd_signature<DROPOUT_FWD_TASK_ID>(),
                forward_task);
}

template <>
OpTaskSignature bwd_signature<DROPOUT_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(fwd_signature<DROPOUT_FWD_TASK_ID>());

  return bwd;
}

template <>
void register_task<DROPOUT_BWD_TASK_ID>() {
  register_task(DROPOUT_BWD_TASK_ID,
                "Dropout Bwd",
                bwd_signature<DROPOUT_BWD_TASK_ID>(),
                backward_task);
}

}; // namespace FlexFlow

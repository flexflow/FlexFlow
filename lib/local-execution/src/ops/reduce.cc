#include "reduce.h"
#include "kernels/reduce_kernels.h"

#include "op-attrs/get_output_shapes.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"
#include "utils/type_traits_core.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Reduce;

enum Slots {
  INPUT,
  OUTPUT,
  ATTRS,
  PROFILING,
  REDUCE,
  PER_DEVICE_STATE,
  HANDLE
};

OpTaskInvocation init(TransposeAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(HANDLE, ff_handle());
  binding.bind_arg(ATTRS, attrs);

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {REDUCE_INIT_TASK_ID, binding};
}

static DeviceSpecific<ReducePerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);
  auto attrs = acc.get_argument<ReduceAttrs>(ATTRS);
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  OperatorType op_type = attrs.op_type;
  // Note: How to set the reduction size?
  size_t reduction_size = input.shape.get_volume() / output.shape.get_volume();
  DeviceSpecific<ReducePerDeviceState> per_device_state =
      init_kernel(handle, op_type, reduction_size, input.shape, output.shape);
  return per_device_state;
}

template <>
void register_task<TRANSPOSE_INIT_TASK_ID>() {
  OpTaskSignature init;
  init.type = OpTaskType::INIT;

  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);
  init.add_arg_slot<ReduceAttrs>(ATTRS);

  init.add_return_value<ReducePerDeviceState>();

  register_task(REDUCE_INIT_TASK_ID, "Reduce::init", init, init_task_impl);
}

// Note: forward_kernel only needs ReducePerDeviceState, input, output
OpTaskInvocation forward(ReduceAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<ReducePerDeviceState>());
  binding.bind_arg(PROFILING, profiling_settings());

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {REDUCE_FWD_TASK_ID, binding};
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_argument<ReducePerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 "[Reduce] forward_time = %.2lfms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr());
}

template <>
void register_task<REDUCE_FWD_TASK_ID>() {
  OpTaskSignature fwd;
  fwd.type = OpTaskType::FWD;

  fwd.add_unchecked_arg_slot<ReducePerDeviceState>(PER_DEVICE_STATE);
  fwd.add_arg_slot<ProfilingSettings>(PROFILING);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  register_task(REDUCE_FWD_TASK_ID, "Reduce::forward", fwd, forward_task_impl);
}

OpTaskInvocation backward(ReduceAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {REDUCE_BWD_TASK_ID, binding};
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_argument<ReducePerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input_grad = acc.get_tensor_grad<Permissions::WO>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 "[Reduce] backward_time = %.2lfms\n",
                 per_device_state,
                 output_grad.get_float_ptr(),
                 input_grad.get_float_ptr());
}

// TODO: OpTaskSignature

// template <>
// void register_task<REDUCE_BWD_TASK_ID>() {
//   OpTaskSignature bwd =
//       infer_bwd_signature(get_op_signature(REDUCE_FWD_TASK_ID));

//   register_task(REDUCE_BWD_TASK_ID, "Reduce::backward", bwd,
//   backward_task_impl);
// }

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ReduceAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view) {
  auto env = sim_factory.new_environment();

  SimTaskBinding init_binding;
  init_binding.bind_arg(ATTRS, attrs);
  init_binding.bind_arg(HANDLE, ff_handle());

  auto init_accessor = env.get_init_accessor(REDUCE_INIT_TASK_ID, init_binding);
  DeviceSpecific<ReducePerDeviceState> per_device_state =
      init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;
  ParallelTensorShape output_shape = get_output_shape(attrs, input.shape);
  fwd_binding.bind(INPUT, input.shape);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind_arg(PER_DEVICE_STATE, per_device_state);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor = env.get_fwd_accessor(REDUCE_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(REDUCE_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

}; // namespace FlexFlow

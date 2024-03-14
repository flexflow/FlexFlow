#include "element_unary.h"
#include "kernels/element_unary_kernels.h"
#include "legion/legion_utilities.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

// declare Legion names
using Legion::Context;
using Legion::PhysicalRegion;
using Legion::Runtime;
using Legion::Task;

using namespace FlexFlow::Kernels::ElementUnary;

enum Slots {
  INPUT,
  INPUT_SHAPE,
  OUTPUT,
  ATTRS,
  HANDLE,
  PROFILING,
  PER_DEVICE_STATE
};

/* ElementUnary */
OpTaskInvocation init(ElementUnaryUnifiedAttrs const &attrs) {
  OpTaskBinding b;

  b.bind_arg(HANDLE, ff_handle());
  b.bind_arg(ATTRS, attrs);
  b.bind_arg(INPUT_SHAPE, input_parallel_tensor_shape(0));

  return {ELEMENTUNARY_INIT_TASK_ID, b};
}

OpTaskInvocation forward(ElementUnaryUnifiedAttrs const &attrs) {
  OpTaskBinding b;

  b.bind(INPUT, input_tensor(0));
  b.bind(OUTPUT, output_tensor(0));

  b.bind_arg(PROFILING, profiling_settings());
  b.bind_arg(PER_DEVICE_STATE,
             per_device_op_state<ElementUnaryPerDeviceState>());

  return {ELEMENTUNARY_FWD_TASK_ID, b};
}

OpTaskInvocation backward(ElementUnaryUnifiedAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {ELEMENTUNARY_BWD_TASK_ID, b};
}

static DeviceSpecific<ElementUnaryPerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {

  auto const &attrs = acc.get_argument<ElementUnaryUnifiedAttrs>(ATTRS);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);
  ParallelTensorShape input_shape =
      acc.get_argument<ParallelTensorShape>(INPUT_SHAPE);
  ParallelTensorShape output_shape = get_output_shape(attrs, input_shape);

  DeviceSpecific<ElementUnaryPerDeviceState> per_device_state =
      acc.create_device_specific<ElementUnaryPerDeviceState>(
          init_kernel(input_shape, output_shape, attrs));
  return per_device_state;
}

static DeviceSpecific<ElementUnaryPerDeviceState>
    init_task(Task const *task,
              std::vector<PhysicalRegion> const &regions,
              Context ctx,
              Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  return init_task_impl(acc);
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto const &attrs = acc.get_argument<ElementUnaryUnifiedAttrs>(ATTRS);

  auto &handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);

  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto per_device_state =
      acc.get_argument<ElementUnaryPerDeviceState>(PER_DEVICE_STATE);

  return profile(forward_kernel,
                 profiling,
                 "[ElementUnary] forward_time = %.2lfms\n",
                 per_device_state,
                 attrs,
                 handle,
                 input,
                 output);
}

static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  forward_task_impl(acc);
}

static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto output = acc.get_tensor<Permissions::RO>(OUTPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);

  auto const &attrs = acc.get_argument<ElementUnaryUnifiedAttrs>(ATTRS);
  auto &handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);

  auto per_device_state =
      acc.get_argument<ElementUnaryPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  return profile(backward_kernel,
                 profiling,
                 "[ElementUnary] backward_time = %.2lfms\n",
                 per_device_state,
                 attrs,
                 handle,
                 input,
                 input_grad,
                 output,
                 output_grad);
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  ElementUnaryUnifiedAttrs const &attrs,
                                  InputParallelTensorDesc const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv) {
  auto env = sim.new_environment();

  ParallelTensorShape output_shape = get_output_shape(attrs, input_shape);

  SimTaskBinding init_binding;
  init_binding.bind_arg(HANDLE, ff_handle());
  init_binding.bind_arg(ATTRS, attrs);
  init_binding.bind_arg(INPUT_SHAPE, input_parallel_tensor_shape(0));

  auto init_accessor =
      env.get_init_accessor(ELEMENTUNARY_INIT_TASK_ID, init_binding);
  DeviceSpecific<ElementUnaryPerDeviceState> per_device_state =
      init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;
  fwd_binding.bind(INPUT, input_shape);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind_arg(PER_DEVICE_STATE, per_device_state);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor =
      env.get_fwd_accessor(ELEMENTUNARY_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor =
      env.get_bwd_accessor(ELEMENTUNARY_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
OpTaskSignature init_signature<ELEMENTUNARY_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);
  init.add_arg_slot<ParallelTensorShape>(INPUT_SHAPE);
  init.add_arg_slot<ElementUnaryUnifiedAttrs>(ATTRS);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  init.add_return_value<ElementUnaryPerDeviceState>();

  return init;
}

template <>
void register_task<ELEMENTUNARY_INIT_TASK_ID>() {
  register_task(ELEMENTUNARY_INIT_TASK_ID,
                "ElementUnary Init",
                init_signature<ELEMENTUNARY_INIT_TASK_ID>(),
                init_task);
}

template <>
OpTaskSignature fwd_signature<ELEMENTUNARY_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_unchecked_arg_slot<ElementUnaryPerDeviceState>(PER_DEVICE_STATE);

  return fwd;
}

template <>
void register_task<ELEMENTUNARY_FWD_TASK_ID>() {
  register_task(ELEMENTUNARY_FWD_TASK_ID,
                "ElementUnary Fwd",
                fwd_signature<ELEMENTUNARY_FWD_TASK_ID>(),
                forward_task);
}

template <>
OpTaskSignature bwd_signature<ELEMENTUNARY_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(fwd_signature<ELEMENTUNARY_FWD_TASK_ID>());

  return bwd;
}

template <>
void register_task<ELEMENTUNARY_BWD_TASK_ID>() {
  register_task(ELEMENTUNARY_BWD_TASK_ID,
                "ElementUnary Bwd",
                bwd_signature<ELEMENTUNARY_BWD_TASK_ID>(),
                backward_task);
}

} // namespace FlexFlow

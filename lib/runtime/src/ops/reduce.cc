#include "reduce.h"
#include "kernels/reduce_kernels.h"
#include "legion/legion_utilities.h"
#include "op-attrs/get_output_shape.h"
#include "utils/exceptions.h"
#include "utils/hash-utils.h"
#include "utils/type_traits_core.h"

namespace FlexFlow {
// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

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

  OperatorType = attrs.op_type;
  // Note: How to set the reduction size?
  size_t reduction_size ;
  DeviceSpecific<ReducePerDeviceState> per_device_state =
      acc.create_device_specific<ReducePerDeviceState>(init_kernel(
          handle, op_type, reduction_size, input.shape, output.shape));
  return per_device_state;
}

static DeviceSpecific<TransposePerDeviceState>
    init_task(Task const *task,
              std::vector<PhysicalRegion> const &regions,
              Context ctx,
              Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  return init_task_impl(acc);
}

template<> void register_task<TRANSPOSE_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT)

  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);
  init.add_arg_slot<ReduceAttrs>(ATTRS);

  init.add_return_value<ReducePerDeviceState>();

  register_task(REDUCE_INIT_TASK_ID, "Reduce::init", init, init_task);
}

// Note: forward_kernel only needs ReducePerDeviceState, input, output
OpTaskInvocation forward(ReduceAttrs const &attrs) {
  OpTaskBinding binding;

  bind.bind_arg(PER_DEVICE_STATE, per_device_op_state<ReducePerDeviceState>());
  bind.bind_arg(PROFILING, profiling_tensor());

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {REDUCE_FWD_TASK_ID, binding};
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
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

static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  forward_task_impl(acc);
}

template <>
void register_task<REDUCE_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FORWARD);

  fwd.add_unchecked_arg_slot<PerDeviceOpState>(PER_DEVICE_STATE);
  fwd.add_arg_slot<ProfilingSettings>(PROFILING);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  register_task(REDUCE_FWD_TASK_ID, "Reduce::forward", fwd, forward_task);
}

OpTaskInvocation backward(ReduceAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {REDUCE_BWD_TASK_ID, binding};
}

static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_argument<ReducePerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input_grad = acc.get_tensor_grad<Permissions::RO>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::WO>(OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 "[Reduce] backward_time = %.2lfms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr());
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

template <>
void register_task<REDUCE_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(get_op_signature(REDUCE_FWD_TASK_ID));

  reister_task(REDUCE_BWD_TASK_ID, "Reduce::backward", bwd, backward_task);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ReduceAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view) {
  auto env = sim.new_environment();

  SimTaskBinding init_binding;
  init_binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(HANDLE, ff_handle());

  auto init_accessor = env.get_init_accessor(REDUCE_INIT_TASK_ID, init_binding);
  DeviceSpecific<ReducePerDeviceState> per_device_state =
      init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;
  ParallelTensorShape output_shape = get_output_shape(attrs, input.shape);
  fwd.bind(INPUT, input.shape);
  fwd.bind(OUTPUT, output_shape);
  fwd.bind_arg(PROFILING, settings);
  fwd.bind_arg(PER_DEVICE_STATE, per_device_state);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor = env.get_fwd_accessor(REDUCE_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(REDUCE_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

}; // namespace FlexFlow

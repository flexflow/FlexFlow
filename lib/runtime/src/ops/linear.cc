#include "linear.h"
#include "kernels/linear_kernels.h"
#include "layer.h"
#include "legion/legion_utilities.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/get_output_shapes.h"
#include "utils/exceptions.h"
#include "utils/graph/views.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::InlineLauncher;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

using namespace FlexFlow::Kernels::Linear;

enum slots {
  INPUT,
  OUTPUT,
  WEIGHT,
  BIAS,
  ATTR,
  PROFILING,
  HANDLE
}; // Note: this needs add more

OpTaskInvocation init(LinearAttrs const &attrs) {
  OpTaskBinding binding;

  bind.bind_arg(HANDLE, ff_handle());
  bind.bind_arg(ATTR, attrs);

  bind.bind(INPUT, input_tensor(0));   // input
  bind.bind(WEIGHT, weight_tensor(0)); // weight
  bind.bind(OUTPUT, output_tensor(0)); // output

  return {LINEAR_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(LinearAttrs const &attrs) {
  OpTaskBinding binding;

  bind.bind(INPUT, input_tensor(0));   // input
  bind.bind(WEIGHT, weight_tensor(0)); // weight
  bind.bind(OUTPUT, output_tensor(0)); // output
  bind.bind(BIAS, bias_tensor(0));     // bias

  bing.bind_arg(PROFILING, profiling_settings());
  bind.bind_arg(PER_DEVICE_STATE, per_device_state<LinearPerDeviceState>());
  bind.bind_arg(ATTRS, attrs);

  return {LINEAR_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(LinearAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {LINEAR_BWD_TASK_ID, b};
}

static DeviceSpecific<LinearPerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto const &attrs = acc.get_argument<MultiHeadAttentionAttrs>(ATTRS);
  Allocator allocator = acc.get_allocator();
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto weight = acc.get_tensor<Permissions::RO>(WEIGHT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  int out_dim = output.shape.at(ff_dim_t{0}) + 1;
  int batch_size = output.shape.get_volume() / out_dim;

  float *one_ptr;

  DeviceSpecific<LinearPerDeviceState> state =
      acc.create_device_specific<LinearPerDeviceState>(
          init_kernel(handle,
                      allocator,
                      one_ptr,
                      attrs.regularizer,
                      attrs.use_bias,
                      input.shape,
                      weight.shape,
                      output.shape,
                      batch_size,
                      attrs.out_channels));
  return state;
}

static DeviceSpecific<MHAPerDeviceState>
    init_task(Task const *task,
              std::vector<PhysicalRegion> const &regions,
              Context ctx,
              Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  return init_task_impl(acc);
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto weight = acc.get_tensor<Permissions::RO>(WEIGHT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto bias = acc.get_tensor<Permissions::RO>(BIAS);

  auto state = acc.get_device_specific<LinearPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto attrs = acc.get_argument<LinearAttrs>(ATTRS);

  int in_dim = input.shape.at(ff_dim_t{0}) + 1;
  int out_dim = output.shape.at(ff_dim_t{0}) + 1;
  int batch_size = output.shape.get_volume() / out_dim;

  float const *bias_ptr = NULL;
  if (attrs.use_bias) {
    bias_ptr = bias.get_float_ptr();
  }

  return profile(forward_kernel,
                 profiling,
                 "[Linear] forward_time = %.2lfms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr(),
                 weight.get_float_ptr(),
                 bias_ptr,
                 in_dim,
                 out_dim,
                 batch_size);
}

static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  forward_task_impl(acc);
};

static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto weight = acc.get_tensor<Permissions::RO>(WEIGHT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto bias = acc.get_tensor<Permissions::RO>(BIAS);

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto weight_grad = acc.get_tensor_grad<Permissions::RW>(WEIGHT);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);
  auto per_device_state = acc.get_argument<MHAPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto attrs = acc.get_argument<LinearAttrs>(ATTRS);

  float const *bias_ptr = NULL;
  if (attrs.use_bias) {
    bias_ptr = bias.get_float_ptr();
  }

  int in_dim = input.shape.at(ff_dim_t{0}) + 1;
  int out_dim = output.shape.at(ff_dim_t{0}) + 1;
  int batch_size = output.shape.get_volume() / out_dim;

  return profile(backward_kernel,
                 profiling,
                 "[Linear] backward_time = %.2lfms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 input_grad.get_float_ptr(),
                 output.get_float_ptr(),
                 output_grad.get_float_ptr(),
                 weight.get_float_ptr(),
                 weight_grad.get_float_ptr(),
                 bias_ptr,
                 in_dim,
                 out_dim,
                 batch_size);
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  LinearAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view) {
  auto env = sim.new_environment();

  ParallelTensorShape output_shape = get_output_shape(input_shape, attrs);

  SimTaskBinding init_binding;
  init_binding.bind(INPUT, input_tensor(0));
  init_binding.bind(WEIGHT, weight_tensor(0));
  init_binding.bind(BIAS, bias_tensor(0));
  init_binding.bind(OUTPUT, output_tensor(0));
  init_binding.bind_arg(ATTRS, attrs);
  init_binding.bind_arg(HANDLE, ff_handle());

  auto init_accessor = env.get_init_accessor(LINEAR_INIT_TASK_ID, init_binding);

  DeviceSpecific<LinearPerDeviceState> per_device_state =
      init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;

  fwd_bind.bind(INPUT, input_tensor(0));   // input
  fwd_bind.bind(WEIGHT, weight_tensor(0)); // weight
  fwd_bind.bind(OUTPUT, output_tensor(0)); // output
  fwd_bind.bind(BIAS, bias_tensor(0));     // bias

  fwd_bid.bind_arg(PROFILING, profiling_settings());
  fwd_bind.bind_arg(PER_DEVICE_STATE, per_device_state<LinearPerDeviceState>());
  fwd_bind.bind_arg(ATTRS, attrs);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor = env.get_accessor(LINEAR_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_accessor(LINEAR_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<LINEAR_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_input_slot(INPUT);
  init.add_input_slot(WEIGHT);
  init.add_input_slot(BIAS);
  init.add_output_slot(OUTPUT);

  init.add_arg_slot<LinearAttrs>(ATTRS);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  init.add_return_value<LinearPerDeviceState>();

  register_task(LINEAR_INIT_TASK_ID, "Linear::init_task", init, init_task);
}

template <>
void register_task<LINEAR_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(INPUT);
  fwd.add_input_slot(WEIGHT);
  fwd.add_input_slot(BIAS);
  fwd.add_output_slot(OUTPUT);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_arg_slot<LinearAttrs>(ATTRS);
  fwd.add_unchecked_arg_slot<MHAPerDeviceState>(PER_DEVICE_STATE);

  register_task(LINEAR_FWD_TASK_ID, "Linear::fwd_task", fwd, forward_task);
}

template <>
void register_task<LINEAR_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(get_op_signature(LINEAR_FWD_TASK_ID));

  register_task(LINEAR_BWD_TASK_ID, "Linear::bwd_task", bwd, backward_task);
}

}; // namespace FlexFlow

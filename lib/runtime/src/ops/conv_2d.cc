#include "conv_2d.h"
#include "kernels/conv_2d_kernels.h"
#include "legion/legion_utilities.h"
#include "mpark/variant.hpp"
#include "op-attrs/get_output_shapes.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

using Legion::Context;
using Legion::PhysicalRegion;
using Legion::Runtime;
using Legion::Task;

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

static DeviceSpecific<Conv2DPerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {

  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);
  auto attrs = acc.get_argument<Conv2DAttrs>(ATTRS);
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto filter = acc.get_tensor<Permissions::RO>(FILTER);
  auto filter_grad = acc.get_tensor_grad<Permissions::RW>(FILTER);

  DeviceSpecific<Conv2DPerDeviceState> per_device_state =
      acc.create_device_specific<Conv2DPerDeviceState>(
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
                      filter_grad.get_float_ptr()));
  return per_device_state;
}

static DeviceSpecific<Conv2DPerDeviceState>
    init_task(Task const *task,
              std::vector<PhysicalRegion> const &regions,
              Context ctx,
              Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  return init_task_impl(acc);
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
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
                 "[Conv2d] forward_time = %.2lfms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr(),
                 filter.get_float_ptr(),
                 bias.get_float_ptr(),
                 attrs.activation);
}

static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  forward_task_impl(acc);
}

static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
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
                 "[Conv2d] backward_time = %.2lfms\n",
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

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  Conv2DAttrs const &attrs,
                                  InputParallelTensorDesc const &input_shape,
                                  InputParallelTensorDesc const &filter_shape,
                                  InputParallelTensorDesc const &bias_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv) {

  auto env = sim.new_environment();

  ParallelTensorShape output_shape = get_output_shape(attrs, input_shape.shape);

  SimTaskBinding init_binding;
  init_binding.bind(INPUT, input_shape);
  init_binding.bind(OUTPUT, output_shape);
  init_binding.bind(FILTER, filter_shape);
  init_binding.bind_arg(ATTRS, attrs);
  init_binding.bind_arg(HANDLE, ff_handle());

  auto init_accessor = env.get_init_accessor(CONV2D_INIT_TASK_ID, init_binding);
  DeviceSpecific<Conv2DPerDeviceState> per_device_state =
      init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind_arg(PER_DEVICE_STATE, per_device_state);
  init_binding.bind_arg(ATTRS, attrs);

  fwd_binding.bind(INPUT, input_shape);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind(FILTER, filter_shape);
  fwd_binding.bind(BIAS, bias_shape);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor = env.get_fwd_accessor(CONV2D_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(CONV2D_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<CONV2D_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_input_slot(INPUT);
  init.add_output_slot(OUTPUT);
  init.add_weight_slot(FILTER);
  init.add_arg_slot<Conv2DAttrs>(ATTRS);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  init.add_return_value<Conv2DPerDeviceState>();

  register_task(CONV2D_INIT_TASK_ID, "Conv2D Init", init, init_task);
}

template <>
void register_task<CONV2D_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<bool>(PROFILING);
  fwd.add_unchecked_arg_slot<Conv2DPerDeviceState>(PER_DEVICE_STATE);
  fwd.add_arg_slot<Conv2DAttrs>(ATTRS);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  fwd.add_weight_slot(FILTER);
  fwd.add_weight_slot(BIAS);

  register_task(CONV2D_FWD_TASK_ID, "Conv2D Fwd", fwd, forward_task);
}

template <>
void register_task<CONV2D_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(get_op_signature(CONV2D_FWD_TASK_ID));

  register_task(CONV2D_BWD_TASK_ID, "Conv2D Bwd", bwd, backward_task);
}

} // namespace FlexFlow

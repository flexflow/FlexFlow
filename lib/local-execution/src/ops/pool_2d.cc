#include "pool_2d.h"
#include "kernels/pool_2d_kernels.h"

#include "op-attrs/get_output_shapes.h"
#include "op-attrs/ops/pool_2d.h"
#include "utils/exception.decl.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"

using namespace FlexFlow::Kernels::Pool2D;

namespace FlexFlow {

enum Slots { INPUT, OUTPUT, ATTRS, PROFILING, PER_DEVICE_STATE, HANDLE };

OpTaskInvocation init(Pool2DAttrs const &attrs) {
  OpTaskBinding binding;
  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));
  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(HANDLE, ff_handle());

  return {POOL2D_INIT_TASK_ID, binding};
}

static DeviceSpecific<Pool2DPerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto const &attrs = acc.get_argument<Pool2DAttrs>(ATTRS);
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  int input_w = input.shape.at(ff_dim_t(0)) + 1;
  int input_h = input.shape.at(ff_dim_t(1)) + 1;
  int input_c = input.shape.at(ff_dim_t(2)) + 1;
  int input_n = input.shape.at(ff_dim_t(3)) + 1;
  int output_w = output.shape.at(ff_dim_t(0)) + 1;
  int output_h = output.shape.at(ff_dim_t(1)) + 1;
  int output_c = output.shape.at(ff_dim_t(2)) + 1;
  int output_n = output.shape.at(ff_dim_t(3)) + 1;

  printf("init pool (input): n(%d) c(%d) h(%d) "
         "w(%d)\n",
         input_n,
         input_c,
         input_h,
         input_w);
  printf("init pool (output): n(%d) c(%d) h(%d) w(%d)\n",
         output_n,
         output_c,
         output_h,
         output_w);

  int pad_h =
      ((output_h - 1) * attrs.stride_h + attrs.kernel_h - input_h + 1) / 2;
  int pad_w =
      ((output_w - 1) * attrs.stride_w + attrs.kernel_w - input_w + 1) / 2;
  if (pad_h != attrs.padding_h) {
    printf("Warning: changing pool_padding_h to satisfy output_h size\n");
  }

  if (pad_w != attrs.padding_w) {
    printf("Warning: changing pool_padding_w to satisfy output_w size\n");
  }

  DeviceSpecific<Pool2DPerDeviceState> state = 
              init_kernel(handle,
                          attrs.activation,
                          input_w,
                          input_h,
                          input_c,
                          input_n,
                          output_w,
                          output_h,
                          output_c,
                          output_n,
                          pad_h,
                          pad_w,
                          attrs.kernel_h,
                          attrs.kernel_w,
                          attrs.stride_h,
                          attrs.stride_w,
                          attrs.pool_type);

  return state;
}


OpTaskInvocation forward(Pool2DAttrs const &attrs) {
  OpTaskBinding binding;
  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<Pool2DPerDeviceState>());

  return {POOL2D_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(Pool2DAttrs const & attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {POOL2D_BWD_TASK_ID, b};
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  Pool2DPerDeviceState state =
      acc.get_argument<Pool2DPerDeviceState>(PER_DEVICE_STATE);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 "[Pool2D] forward_time = %.2lfms\n",
                 state,
                 input.get_float_ptr(),
                 output.get_float_ptr());
}



static std::optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  Pool2DPerDeviceState state =
      acc.get_argument<Pool2DPerDeviceState>(PER_DEVICE_STATE);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto input_grad = acc.get_tensor<Permissions::RW>(INPUT);
  auto output = acc.get_tensor<Permissions::RO>(OUTPUT);
  auto output_grad = acc.get_tensor<Permissions::RO>(OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 "[Pool2D] backward_time = %.2lfms\n",
                 state,
                 input.get_float_ptr(),
                 input_grad.get_float_ptr(),
                 output.get_float_ptr(),
                 output_grad.get_float_ptr());
}



CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  Pool2DAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view) {
  auto env = sim_factory.new_environment();
  ParallelTensorShape output_shape = get_output_shape(attrs, input.shape);

  SimTaskBinding init_binding;
  init_binding.bind(INPUT, input.shape);
  init_binding.bind(OUTPUT, output_shape);
  init_binding.bind_arg(ATTRS, attrs);
  init_binding.bind_arg(HANDLE, ff_handle());

  auto init_accessor = env.get_init_accessor(POOL2D_INIT_TASK_ID, init_binding);

  DeviceSpecific<Pool2DPerDeviceState> per_device_state =
      init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;

  fwd_binding.bind(INPUT, input.shape);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind_arg(PER_DEVICE_STATE, per_device_state);

  auto fwd_accessor = env.get_fwd_accessor(POOL2D_FWD_TASK_ID, fwd_binding);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto bwd_accessor = env.get_bwd_accessor(POOL2D_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);

  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<POOL2D_INIT_TASK_ID>() {
  OpTaskSignature init; init.type = OpTaskType::INIT;

  init.add_input_slot(INPUT);
  init.add_output_slot(OUTPUT);

  init.add_arg_slot<Pool2DAttrs>(ATTRS);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  init.add_return_value<FlexFlow::Pool2DPerDeviceState>();

  register_task(POOL2D_INIT_TASK_ID, "Pool2D::init", init, init_task_impl);
}

template <>
void register_task<POOL2D_FWD_TASK_ID>() {
  OpTaskSignature fwd; fwd.type = OpTaskType::FWD;

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  fwd.add_arg_slot<ProfilingSettings>(PROFILING);

  fwd.add_arg_slot<Pool2DPerDeviceState>(PER_DEVICE_STATE);

  register_task(POOL2D_FWD_TASK_ID, "Pool2D::forward", fwd, forward_task_impl);
}

// TODO: OpTaskSignature

// template <>
// void register_task<POOL2D_BWD_TASK_ID>() {
//   OpTaskSignature bwd =
//       infer_bwd_signature(get_op_signature(POOL2D_FWD_TASK_ID));

//   register_task(POOL2D_BWD_TASK_ID, "Pool2D::backward", bwd, backward_task_impl);
// }

}; // namespace FlexFlow

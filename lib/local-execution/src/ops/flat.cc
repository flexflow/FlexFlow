#include "flat.h"
#include "kernels/flat_kernels.h"
#include "op-attrs/get_output_shapes.h"

namespace FlexFlow {

// declare Legion names





using namespace FlexFlow::Kernels::Flat;

enum SLOTS { INPUT, OUTPUT, HANDLE, PROFILING };

OpTaskInvocation forward(FlatAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  binding.bind_arg(PROFILING, profiling_settings());
  return {FLAT_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(FlatAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {FLAT_BWD_TASK_ID, b};
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 "[Flat] forward_time = %.2lfms\n",
                 input,
                 output.get_float_ptr());
}



static std::optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 "[Flat] forward_time = %.2lfms\n",
                 input,
                 input_grad.get_float_ptr(),
                 output_grad.get_float_ptr());
}



CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  FlatAttrs const &attrs,
                                  InputParallelTensorDesc const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv) {
  auto env = sim.new_environment();

  ParallelTensorShape output_shape = get_output_shape(attrs, input_shape.shape);

  SimTaskBinding fwd_binding;
  fwd_binding.bind(INPUT, input_shape);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind_arg(PROFILING, settings);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor = env.get_fwd_accessor(FLAT_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(FLAT_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
OpTaskSignature fwd_signature<FLAT_FWD_TASK_ID>() {
  OpTaskSignature fwd; fwd.type = OpTaskType::FWD;

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  return fwd;
}

template <>
void register_task<FLAT_FWD_TASK_ID>() {
  register_task(FLAT_FWD_TASK_ID,
                "Flat Fwd",
                fwd_signature<FLAT_FWD_TASK_ID>(),
                forward_task_impl);
}

template <>
OpTaskSignature bwd_signature<FLAT_BWD_TASK_ID>() {
  OpTaskSignature bwd = infer_bwd_signature(fwd_signature<FLAT_FWD_TASK_ID>());

  return bwd;
}

template <>
void register_task<FLAT_BWD_TASK_ID>() {
  register_task(FLAT_BWD_TASK_ID,
                "Flat Bwd",
                bwd_signature<FLAT_BWD_TASK_ID>(),
                backward_task_impl);
}

}; // namespace FlexFlow

#include "flat.h"
#include "kernels/flat_kernels.h"
#include "op-attrs/get_output_shapes.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Flat;

enum SLOTS { INPUT, OUTPUT, HANDLE, PROFILING };

OpTaskInvocation forward(FlatAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  binding.bind_arg(PROFILING, profiling_settings());
  return {task_id_t::FLAT_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(FlatAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {task_id_t::FLAT_BWD_TASK_ID, b};
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 "[Flat] forward_time = {:.2lf}ms\n",
                 input,
                 output.get_float_ptr());
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 "[Flat] backward_time = {:.2lf}ms\n",
                 input,
                 input_grad.get_float_ptr(),
                 output_grad.get_float_ptr());
}

TaskImplFunction get_flat_fwd_task_impl() {
  return TaskImplFunction{FwdBwdTaskImplFunction{forward_task_impl}};
}
TaskImplFunction get_flat_bwd_task_impl() {
  return TaskImplFunction{FwdBwdTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_flat_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  return fwd;
}

OpTaskSignature get_flat_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_flat_fwd_signature());

  return bwd;
}

std::vector<task_id_t> get_task_ids(FlatAttrs const &) {
  return {task_id_t::FLAT_FWD_TASK_ID, task_id_t::FLAT_BWD_TASK_ID};
}

}; // namespace FlexFlow

#include "dropout.h"
#include "kernels/dropout_kernels.h"
#include "local-execution/op_task_invocation.h"
#include "local-execution/op_task_signature.h"
#include "op-attrs/get_output_shapes.h"
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

static DeviceSpecificDeviceStates
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  Allocator allocator = acc.get_allocator();
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(FF_HANDLE);
  auto const &attrs = acc.get_argument<DropoutAttrs>(ATTRS);

  DropoutPerDeviceState per_device_state =
      init_kernel(handle, attrs.rate, attrs.seed, output.shape, allocator);
  return DeviceSpecificDeviceStates{
      DeviceSpecific<DropoutPerDeviceState>::create(per_device_state)};
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_argument<DropoutPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 "[Dropout] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr());
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto const &attrs = acc.get_argument<DropoutAttrs>(ATTRS);
  auto per_device_state =
      acc.get_argument<DropoutPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 "[Dropout] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 output_grad.get_float_ptr(),
                 input_grad.get_float_ptr());
}

TaskImplFunction get_dropout_init_task_impl() {
  return TaskImplFunction{init_task_impl};
}
TaskImplFunction get_dropout_fwd_task_impl() {
  return TaskImplFunction{forward_task_impl};
}
TaskImplFunction get_dropout_bwd_task_impl() {
  return TaskImplFunction{backward_task_impl};
}

OpTaskSignature get_dropout_init_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_arg_slot<DropoutAttrs>(ATTRS);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(FF_HANDLE);
  init.add_output_slot(OUTPUT);

  init.add_return_value<DropoutPerDeviceState>();

  return init;
}

OpTaskSignature get_dropout_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_unchecked_arg_slot<DropoutPerDeviceState>(PER_DEVICE_STATE);
  fwd.add_arg_slot<ProfilingSettings>(PROFILING);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  return fwd;
}

OpTaskSignature get_dropout_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_dropout_fwd_signature());

  return bwd;
}

std::vector<task_id_t> get_task_ids(DropoutAttrs const &) {
  return {DROPOUT_INIT_TASK_ID, DROPOUT_FWD_TASK_ID, DROPOUT_BWD_TASK_ID};
}

}; // namespace FlexFlow

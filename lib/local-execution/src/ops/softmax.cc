/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "softmax.h"
#include "kernels/softmax_kernels.h"
#include "op-attrs/get_output_shapes.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"

namespace FlexFlow {
// declare Legion names






using namespace FlexFlow::Kernels::Softmax;

enum Slots { INPUT, OUTPUT, ATTRS, PROFILING, PER_DEVICE_STATE, HANDLE };

OpTaskInvocation init(SoftmaxAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(HANDLE, ff_handle());
  binding.bind_arg(ATTRS, attrs);
  return {SOFTMAX_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(SoftmaxAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<SoftmaxPerDeviceState>());
  binding.bind_arg(PROFILING, profiling_settings());

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {SOFTMAX_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(SoftmaxAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {SOFTMAX_BWD_TASK_ID, binding};
}

static DeviceSpecific<SoftmaxPerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);

  auto const &attrs = acc.get_argument<SoftmaxAttrs>(ATTRS);

  DeviceSpecific<SoftmaxPerDeviceState> per_device_state =
          init_kernel(handle, attrs.dim);
  return per_device_state;
}


static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto per_device_state =
      acc.get_argument<SoftmaxPerDeviceState>(PER_DEVICE_STATE);

  return profile(forward_kernel,
                 profiling,
                 "[SoftMax] forward_time = %.2lfms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr(), );
}



static std::optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  assert(input_grad.shape == input.shape);

  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);
  auto output = acc.get_tensor<Permissions::RO>(OUTPUT);

  assert(output_grad.shape == output.shape);

  return profile(
      backward_kernel,
      profiling,
      "[SoftMax] backward_time = %.2lfms\n",
      input_grad.get_float_ptr(),
      output_grad.get_float_ptr(),
      output_grad.shape.volume(), // Note(lambda): get num_elements, maybe wrong
  );
}



CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  SoftmaxAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view) {
  auto env = sim.new_environment();

  ParallelTensorShape output_shape = get_output_shape(attrs, input.shape);

  SimTaskBinding init_binding;

  init_binding.bind_arg(ATTRS, attrs);
  init_binding.bind_arg(HANDLE, ff_handle());

  auto init_accessor =
      env.get_init_accessor(SOFTMAX_INIT_TASK_ID, init_binding);
  DeviceSpecific<SoftmaxPerDeviceState> per_device_state =
      init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;
  ParallelTensorShape output_shape = get_output_shape(attrs, input.shape);
  fwd_binding.bind(INPUT, input.shape);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind_arg(PER_DEVICE_STATE, per_device_state);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor = env.get_fwd_accessor(SOFTMAX_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(SOFTMAX_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<SOFTMAX_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);
  init.add_arg_slot<SoftmaxAttrs>(ATTRS);
  init.add_return_value_slot<SoftmaxPerDeviceState>();

  register_task(SOFTMAX_INIT_TASK_ID, "SoftMax Init", init, init_task);
}

template <>
void register_task<SOFTMAX_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_unchecked_arg_slot<SoftmaxPerDeviceState>(PER_DEVICE_STATE);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  register_task(SOFTMAX_FWD_TASK_ID, "SoftMax Fwd", fwd, forward_task);
}

template <>
void register_task<SOFTMAX_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(get_op_signature(SOFTMAX_FWD_TASK_ID));

  register_task(SOFTMAX_BWD_TASK_ID, "SoftMax Bwd", bwd, backward_task);
}

}; // namespace FlexFlow

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

#include "reshape.h"
#include "kernels/reshape_kernels.h"
#include "op-attrs/get_output_shapes.h"

namespace FlexFlow {
// declare Legion names

using namespace FlexFlow::Kernels::Reshape;

enum slots { INPUT, OUTPUT, ATTRS, PROFILING, PER_DEVICE_STATE };

OpTaskInvocation init(ReshapeAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(ATTRS, attrs);

  return {RESHAPE_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(ReshapeAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<ReshapePerDeviceState>());
  binding.bind_arg(PROFILING, profiling_settings());

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));
  return {RESHAPE_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(ReshapeAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {RESHAPE_BWD_TASK_ID, binding};
}

static DeviceSpecific<ReshapePerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto attrs = acc.get_argument<ReshapeAttrs>(ATTRS);

  ReshapePerDeviceState per_device_state = init_kernel(attrs.shape.data_type);
  return DeviceSpecific<ReshapePerDeviceState>::create(per_device_state);
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_argument<ReshapePerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 "[Reshape] forward time = {:.2lf}ms\n",
                 per_device_state,
                 input,
                 output);
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_argument<ReshapePerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 "[Reshape] backward time = {:.2lf}ms\n",
                 per_device_state,
                 input_grad,
                 output_grad);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ReshapeAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view) {

  auto env = sim_factory.new_environment();
  SimTaskBinding init_binding;
  init_binding.bind_arg(ATTRS, attrs);
  auto init_accessor =
      env.get_init_accessor(RESHAPE_INIT_TASK_ID, init_binding);
  auto per_device_state = init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;
  ParallelTensorShape output_shape = get_output_shape(attrs, input.shape);
  fwd_binding.bind_arg(PER_DEVICE_STATE, per_device_state);
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind(INPUT, input.shape);
  fwd_binding.bind(OUTPUT, output_shape);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor = env.get_fwd_accessor(RESHAPE_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(RESHAPE_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<RESHAPE_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_arg_slot<ReshapeAttrs>(ATTRS);

  init.add_return_value<ReshapePerDeviceState>();

  register_task(RESHAPE_INIT_TASK_ID, "Reshape Init", init, init_task_impl);
}

template <>
void register_task<RESHAPE_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_unchecked_arg_slot<ReshapePerDeviceState>(PER_DEVICE_STATE);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  register_task(RESHAPE_FWD_TASK_ID, "Reshape Fwd", fwd, forward_task_impl);
}

// TODO: OpTaskSignature

// template <>
// void register_task<RESHAPE_BWD_TASK_ID>() {
//   OpTaskSignature bwd =
//       infer_bwd_binding(get_op_signature(RESHAPE_FWD_TASK_ID));

//   register_task(RESHAPE_BWD_TASK_ID, "Reshape Bwd", bwd, backward_task_impl);
// }

}; // namespace FlexFlow

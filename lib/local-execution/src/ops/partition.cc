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

#include "kernels/partition_kernels.h"
#include "op-attrs/get_output_shapes.h"
#include "repartition.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Repartition;

enum Slots { INPUT, OUTPUT, ATTRS, PROFILING, HANDLE, PER_DEVICE_STATE };

OpTaskInvocation init(RepartitionAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(HANDLE, ff_handle());
  binding.bind(INPUT, input_tensor(0));

  return {REPARTITION_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(RepartitionAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<RepartitionPerDeviceState>());
  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {REPARTITION_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(RepartitionAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {REPARTITION_BWD_TASK_ID, binding};
}

static DeviceSpecific<RepartitionPerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);

  // Note: use the input data type

  // TODO: refactor ops as follows
  // DeviceSpecific<RepartitionPerDeviceState> per_device_state = init_kernel(handle, input.data_type);

  RepartitionPerDeviceState per_device_state =
      init_kernel(handle, input.data_type);
  return DeviceSpecific<RepartitionPerDeviceState>::create(per_device_state);
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto per_device_state =
      acc.get_argument<RepartitionPerDeviceState>(PER_DEVICE_STATE);
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 "[Reparition/Partition] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input,
                 output);
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto per_device_state =
      acc.get_argument<RepartitionPerDeviceState>(PER_DEVICE_STATE);
  auto input_grad = acc.get_tensor_grad<Permissions::RO>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::WO>(OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 "[Reparition/Partition] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 output_grad,
                 input_grad);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  RepartitionAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view) {
  auto env = sim_factory.new_environment();

  ParallelTensorShape output_shape = get_output_shape(attrs, input.shape);

  SimTaskBinding init_binding;
  init_binding.bind_arg(HANDLE, ff_handle());
  init_binding.bind(INPUT, input.shape); // use the input data type
  auto init_accessor =
      env.get_init_accessor(REPARTITION_INIT_TASK_ID, init_binding);

  DeviceSpecific<RepartitionPerDeviceState> per_device_state =
      init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;
  fwd_binding.bind(INPUT, input.shape);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind_arg(PER_DEVICE_STATE, per_device_state);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor =
      env.get_fwd_accessor(REPARTITION_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor =
      env.get_bwd_accessor(REPARTITION_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<REPARTITION_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  init.add_input_slot(INPUT);

  init.add_return_value<RepartitionPerDeviceState>();

  register_task(
      REPARTITION_INIT_TASK_ID, "Repartition Init", init, init_task_impl);
}

template <>
void register_task<REPARTITION_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_unchecked_arg_slot<RepartitionPerDeviceState>(PER_DEVICE_STATE);

  register_task(
      REPARTITION_FWD_TASK_ID, "Repartition Fwd", fwd, forward_task_impl);
}

// TODO: OpTaskSignature

// template <>
// void register_task<REPARTITION_BWD_TASK_ID>() {
//   OpTaskSignature bwd =
//       infer_bwd_signature(get_op_signature(REPARTITION_FWD_TASK_ID));

//   register_task(REPARTITION_BWD_TASK_ID, "Repartition Bwd", bwd,
//   backward_task_impl);
// }

}; // namespace FlexFlow

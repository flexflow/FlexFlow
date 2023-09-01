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

#include "combine.h"
#include "kernels/combine_kernels.h"
#include "utils/hash-utils.h"

namespace FlexFlow {
// declare Legion names
using Legion::Context;
using Legion::PhysicalRegion;
using Legion::Runtime;
using Legion::Task;

using namespace FlexFlow::Kernels::Combine;

enum Slots { INPUT, OUTPUT, PROFILING, PER_DEVICE_STATE };

OpTaskInvocation init(CombineAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(INPUT, input_tensor(0));

  return {COMBINE_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(CombineAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<CombinePerDeviceState>());
  binding.bind_arg(PROFILING, profiling_settings());

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {COMBINE_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(CombineAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {COMBINE_BWD_TASK_ID, b};
}

static DeviceSpecific<CombinePerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {

  auto input = acc.get_tensor<Permissions::RO>(INPUT);

  DeviceSpecific<CombinePerDeviceState> per_device_state =
      acc.create_device_specific<CombinePerDeviceState>(
          init_kernel(input.data_type));
  return per_device_state;
}

static DeviceSpecific<CombinePerDeviceState>
    init_task(Task const *task,
              std::vector<PhysicalRegion> const &regions,
              Context ctx,
              Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  return init_task_impl(acc);
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_argument<CombinePerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 "[Combine] forward_time = %.2lfms\n",
                 &per_device_state,
                 input,
                 output);
}

static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  forward_task_impl(acc);
}

static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_argument<CombinePerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input_grad = acc.get_tensor_grad<Permissions::RO>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::WO>(OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 "[Combine] forward_time = %.2lfms\n",
                 &per_device_state,
                 input_grad,
                 output_grad);
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  CombineAttrs const &attrs,
                                  InputParallelTensorDesc const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv) {
  auto env = sim.new_environment();
  // TODO: to be implemented
  float forward_time = 0.5;
  float backward_time = 0.5;
  float sync_time = 0.0;
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<COMBINE_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_input_slot(INPUT);

  register_task(COMBINE_INIT_TASK_ID, "Combine Init", init, init_task);
}

template <>
void register_task<COMBINE_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<bool>(PROFILING);
  fwd.add_unchecked_arg_slot<CombinePerDeviceState>(PER_DEVICE_STATE);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  register_task(COMBINE_FWD_TASK_ID, "Combine Fwd", fwd, forward_task);
}

template <>
void register_task<COMBINE_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(get_op_signature(COMBINE_FWD_TASK_ID));

  register_task(COMBINE_BWD_TASK_ID, "Combine Bwd", bwd, backward_task);
}

}; // namespace FlexFlow

namespace std {}; // namespace std

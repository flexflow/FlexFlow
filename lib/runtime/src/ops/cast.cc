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

#include "cast.h"
#include "kernels/cast_kernels.h"
#include "legion/legion_utilities.h"
#include "utils/hash-utils.h"

using namespace FlexFlow::Kernels::Cast;

using Legion::Context;
using Legion::PhysicalRegion;
using Legion::Runtime;
using Legion::Task;

namespace FlexFlow {

enum Slots { INPUT, OUTPUT, ATTRS, PROFILING, PER_DEVICE_STATE, HANDLE };

OpTaskInvocation init(CastAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(HANDLE, ff_handle());

  return {CAST_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(CastAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PER_DEVICE_STATE, per_device_op_state<CastPerDeviceState>());
  binding.bind_arg(PROFILING, profiling_settings());
  b.bind_arg(ATTRS, attrs);

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {CAST_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(CastAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {CAST_BWD_TASK_ID, binding};
}

static DeviceSpecific<CastPerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {

  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);

  DeviceSpecific<CastPerDeviceState> per_device_state =
      acc.create_device_specific<CastPerDeviceState>(init_kernel(handle));
  return per_device_state;
}

static DeviceSpecific<CastPerDeviceState>
    init_task(Task const *task,
              std::vector<PhysicalRegion> const &regions,
              Context ctx,
              Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  return init_task_impl(acc);
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_argument<CastPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto const &attrs = acc.get_argument<CastAttrs>(ATTRS);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 "[Cast] forward_time = %.2lfms\n",
                 &per_device_state,
                 input,
                 output,
                 input.data_type,
                 attrs.dtype);
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
      acc.get_argument<CastPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto const &attrs = acc.get_argument<CastAttrs>(ATTRS);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);

  auto input_grad = acc.get_tensor_grad<Permissions::RO>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::WO>(OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 "[Cast] forward_time = %.2lfms\n",
                 &per_device_state,
                 input_grad,
                 output_grad,
                 input.data_type,
                 attrs.dtype);
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  CastAttrs const &attrs,
                                  InputParallelTensorDesc const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv) {
  auto env = sim.new_environment();

  // Assume cast has no cost
  float forward_time = 0.0;
  float backward_time = 0.0;
  float sync_time = 0.0;
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<CAST_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);
  init.add_return_value<CastPerDeviceState>();

  register_task(CAST_INIT_TASK_ID, "Cast Init", init, init_task);
}

template <>
void register_task<CAST_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<CastAttrs>(ATTRS);
  fwd.add_arg_slot<bool>(PROFILING);
  fwd.add_unchecked_arg_slot<CastPerDeviceState>(PER_DEVICE_STATE);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  register_task(CAST_FWD_TASK_ID, "Cast Fwd", fwd, forward_task);
}

template <>
void register_task<CAST_BWD_TASK_ID>() {
  OpTaskSignature bwd = infer_bwd_signature(get_op_signature(CAST_FWD_TASK_ID));

  register_task(CAST_BWD_TASK_ID, "Cast Bwd", bwd, backward_task);
}

}; // namespace FlexFlow

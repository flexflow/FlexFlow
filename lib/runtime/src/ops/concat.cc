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

#include "concat.h"
#include "kernels/concat_kernels.h"
#include "legion/legion_utilities.h"
#include "task_spec/variadic_tensor_ref.h"
#include "utils/hash-utils.h"
#include "op-attrs/get_output_shapes.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Concat;

using Legion::Context;
using Legion::PhysicalRegion;
using Legion::Runtime;
using Legion::Task;

enum Slots {
  INPUTS,
  OUTPUT,
  ATTRS,
  PROFILING,
  HANDLE,
  PER_DEVICE_STATE,
  NUM_INPUTS
};

OpTaskInvocation init(ConcatAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(ATTRS, attrs);

  return {CONCAT_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(ConcatAttrs const &attrs) {
  OpTaskBinding binding;
  binding.bind_arg(PER_DEVICE_STATE, per_device_op_state<ConcatPerDeviceState>());
  binding.bind(INPUTS, get_input_tensors());
  binding.bind(OUTPUT, output_tensor(0));
  binding.bind(NUM_INPUTS, get_number_inputs());
  binding.bind_arg(PROFILING, profiling_settings());

  return {CONCAT_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(ConcatAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {CONCAT_BWD_TASK_ID, b};
}

static DeviceSpecific<ConcatPerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto const &attrs = acc.get_argument<ConcatAttrs>(ATTRS);
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);

    DeviceSpecific<ConcatPerDeviceState> per_device_state =
      acc.create_device_specific<ConcatPerDeviceState>(init_kernel(attrs.axis));
  return per_device_state;
}

static DeviceSpecific<ConcatPerDeviceState>
    init_task(Task const *task,
              std::vector<PhysicalRegion> const &regions,
              Context ctx,
              Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  return init_task_impl(acc);
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state = acc.get_argument<ConcatPerDeviceState>(PER_DEVICE_STATE);
  int number_of_inputs = acc.get_argument<int>(NUM_INPUTS);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto inputs = acc.get_variadic_tensor<Permissions::RO>(INPUTS);

  return profile(forward_kernel,
          profiling,
          "[Concat] forward_time = %.2lfms\n",
          &per_device_state,
          output,
          inputs,
          number_of_inputs);
}

static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  forward_task_impl(acc);
}

static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state = acc.get_argument<ConcatPerDeviceState>(PER_DEVICE_STATE);
  int number_of_inputs = acc.get_argument<int>(NUM_INPUTS);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input_grads = acc.get_variadic_tensor_grad<Permissions::RW>(INPUTS);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);

  assert(number_of_inputs <= MAX_NUM_INPUTS);

  return profile(backward_kernel,
          profiling,
          "[Concat] backward_time = %.2lfms\n",
          &per_device_state,
          output_grad,
          input_grads,
          number_of_inputs);
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  ConcatAttrs const &attrs,
                                  InputVariadicParallelTensorDesc const &inputs_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv) {
  int numInputs = (inputs_shape.shapes).size();
  assert(numInputs <= MAX_NUM_INPUTS);

  auto env = sim.new_environment();

  ParallelTensorShape output_shape = get_output_shape(attrs, inputs_shape.shapes);

  SimTaskBinding init_binding;
  init_binding.bind_arg(PROFILING, settings);
  init_binding.bind_arg(ATTRS, attrs);

  auto init_accessor =
      env.get_init_accessor(CONCAT_INIT_TASK_ID, init_binding);
  DeviceSpecific<ConcatPerDeviceState> per_device_state =
      init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;
  fwd_binding.bind_arg(PER_DEVICE_STATE, per_device_state);
  fwd_binding.bind(INPUTS, inputs_shape);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind_arg(NUM_INPUTS, numInputs);
  fwd_binding.bind_arg(PROFILING, settings);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor = env.get_fwd_accessor(CONCAT_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(CONCAT_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);

}

template <>
void register_task<CONCAT_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_arg_slot<ConcatAttrs>(ATTRS);
  init.add_arg_slot<bool>(PROFILING);
  
  register_task(CONCAT_INIT_TASK_ID, "Concat Init", init, init_task);
}

template <>
void register_task<CONCAT_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<int>(NUM_INPUTS);
  fwd.add_arg_slot<bool>(PROFILING);
  fwd.add_input_slot(INPUTS, SlotType::VARIADIC);
  fwd.add_output_slot(OUTPUT);
  fwd.add_unchecked_arg_slot<ConcatPerDeviceState>(PER_DEVICE_STATE);

  register_task(CONCAT_FWD_TASK_ID, "Concat Fwd", fwd, forward_task);
}

template <>
void register_task<CONCAT_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(get_op_signature(CONCAT_FWD_TASK_ID));

  register_task(CONCAT_BWD_TASK_ID, "BatchMatmul Bwd", bwd, backward_task);
}


}; // namespace FlexFlow

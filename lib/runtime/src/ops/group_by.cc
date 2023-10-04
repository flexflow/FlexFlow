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

#include "groupby.h"
#include "kernels/groupby_kernels.h"
#include "legion/legion_utilities.h"
#include "op-attrs/get_output_shapes.h"
#include "utils/hash-utils.h"

namespace FlexFlow {
// declare Legion names
using Legion::Context;
using Legion::PhysicalRegion;
using Legion::Runtime;
using Legion::Task;

using namespace FlexFlow::Kernels::GroupBy;

enum Slots { INPUT, OUTPUT, ASSIGN, ATTRS, PROFILING, PER_DEVICE_STATE };

OpTaskInvocation init(Group_byAttrs const &attrs) {
  OpTaskBinding binding;
  binding.bind_arg(ATTRS, attrs);

  return {GROUP_BY_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(Group_byAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<GroupByPerDeviceState>());

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));
  binding.bind(ASSIGN, weight_tensor(0));

  return {GROUP_BY_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(Group_byAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {GROUP_BY_BWD_TASK_ID, binding};
}

static DeviceSpecific<GroupByPerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto attrs = acc.get_argument<Group_byAttrs>(ATTRS);

  DeviceSpecific<GroupByPerDeviceState> per_device_state =
      acc.create_device_specific<GroupByPerDeviceState>(init_kernel(attrs.n));
  return per_device_state;
}

static DeviceSpecific<GroupByPerDeviceState>
    init_task(Task const *task,
              std::vector<PhysicalRegion> const &regions,
              Context ctx,
              Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  return init_task_impl(acc);
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto per_device_state =
      acc.get_argument<GroupByPerDeviceState>(PER_DEVICE_STATE);
  auto attrs = acc.get_argument<Group_byAttrs>(ATTRS);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto assign = acc.get_tensor<Permissions::RO>(ASSIGN);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 "[Group By] forward_time = %.2lfms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 assign.get_int32_ptr(),
                 output.get_float_ptr(),
                 attrs.n,
                 input.shape[legion_dim_t(0)], // int k
                 attrs.alpha,
                 input.shape.get_volume(), // batch_size
                 input.shape.get_dim()     // data_dim
  );
}

static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  forward_task_impl(acc);
}

static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto per_device_state =
      acc.get_argument<GroupByPerDeviceState>(PER_DEVICE_STATE);
  auto attrs = acc.get_argument<Group_byAttrs>(ATTRS);

  auto input = acc.get_tensor<Permissions::WO>(INPUT);
  auto input_grad = acc.get_tensor_grad<Permissions::WO>(INPUT);
  auto assign = acc.get_tensor<Permissions::RO>(ASSIGN);
  auto output_grad = acc.get_tensor_grad<Permissions::WO>(OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 "[Group By] backward_time = %.2lfms\n",
                 per_device_state,
                 input_grad.get_float_ptr(),
                 assign.get_int32_ptr(),
                 output_grad.get_float_ptr(),
                 attrs.n,
                 input.shape[legion_dim_t(0)], // int k
                 attrs.alpha,
                 input.shape.get_volume(), // batch_size
                 input.shape.get_dim()     // data_dim
  );
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  Group_byAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  InputParallelTensorDesc const &assign,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv) {

  auto env = sim.new_environment();

  ParallelTensorShape output_shape =
      get_output_shape(attrs, input.shape, assign.shape);

  SimTaskBinding init_binding;
  init_binding.bind_arg(ATTRS, attrs);

  auto init_accessor =
      env.get_init_accessor(GROUP_BY_INIT_TASK_ID, init_binding);
  DeviceSpecific<GroupByPerDeviceState> per_device_state =
      init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind_arg(PER_DEVICE_STATE, per_device_state);
  init_binding.bind_arg(ATTRS, attrs);

  fwd_binding.bind(INPUT, input);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind(ASSIGN, assign);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor = env.get_fwd_accessor(GROUP_BY_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(GROUP_BY_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
OpTaskSignature init_signature<GROUP_BY_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);
  init.add_arg_slot<Group_byAttrs>(ATTRS);

  init.add_return_value<GroupByPerDeviceState>();

  return init;
}

template <>
void register_task<GROUP_BY_INIT_TASK_ID>() {
  register_task(GROUP_BY_INIT_TASK_ID,
                "Group By init",
                init_signature<GROUP_BY_INIT_TASK_ID>(),
                init_task);
}

template <>
OpTaskSignature fwd_signature<GROUP_BY_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<bool>(PROFILING);
  fwd.add_unchecked_arg_slot<GroupByPerDeviceState>(PER_DEVICE_STATE);
  fwd.add_arg_slot<Group_byAttrs>(ATTRS);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  fwd.add_weight_slot(ASSIGN);

  return fwd;
}

template <>
void register_task<GROUP_BY_FWD_TASK_ID>() {
  register_task(GROUP_BY_FWD_TASK_ID,
                "Group By Fwd",
                fwd_signature<GROUP_BY_FWD_TASK_ID>(),
                forward_task);
}

template <>
OpTaskSignature bwd_signature<GROUP_BY_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(get_op_signature(GROUP_BY_FWD_TASK_ID));
  return bwd;
}

template <>
void register_task<GROUP_BY_BWD_TASK_ID>() {
  register_task(GROUP_BY_BWD_TASK_ID,
                "Group By Bwd",
                bwd_signature<GROUP_BY_BWD_TASK_ID>(),
                BATCHNORM_FWD_TASK_ID);
}

}; // namespace FlexFlow

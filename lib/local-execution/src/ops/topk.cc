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

#include "topk.h"
#include "kernels/topk_kernels.h"
#include "op-attrs/get_output_shapes.h"
#include "utils/exception.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::TopK;

// For an input tensor, computes the top k entries in each row
// (resp. vector along the last dimension). Thus,
// values.shape = indices.shape = input.shape[:-1] + [k]

enum Slots { INPUT, OUTPUT, INDICES, ATTRS, PROFILING, PER_DEVICE_STATE };

OpTaskInvocation init(TopKAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(ATTRS, attrs);

  return {TOPK_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(TopKAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PER_DEVICE_STATE, per_device_op_state<TopKPerDeviceState>());
  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(ATTRS, attrs);

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));
  binding.bind(INDICES, output_tensor(1));

  return {TOPK_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(TopKAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {TOPK_BWD_TASK_ID, binding};
}

static DeviceSpecific<DeviceStates>
    init_task_impl(TaskArgumentAccessor const &acc) {

  auto attrs = acc.get_argument<TopKAttrs>(ATTRS);

  TopKPerDeviceState per_device_state = init_kernel(attrs.sorted);
  return DeviceSpecific<DeviceStates>::create(per_device_state);
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto attrs = acc.get_argument<TopKAttrs>(ATTRS);
  auto per_device_state =
      acc.get_argument<TopKPerDeviceState>(PER_DEVICE_STATE);
  auto profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  int length = input.shape.at(legion_dim_t(0)) + 1;
  size_t batch_size = input.shape.get_volume() / length;
  auto indices = acc.get_tensor<Permissions::WO>(INDICES);

  return profile(forward_kernel,
                 profiling,
                 "[TopK] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr(),
                 indices.get_int32_ptr(),
                 batch_size,
                 length,
                 attrs.k,
                 attrs.sorted);
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto attrs = acc.get_argument<TopKAttrs>(ATTRS);
  auto per_device_state =
      acc.get_argument<TopKPerDeviceState>(PER_DEVICE_STATE);
  auto profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);

  auto indices = acc.get_tensor<Permissions::RO>(INDICES);

  int length = input_grad.shape.at(legion_dim_t(0)) + 1;
  size_t batch_size = input_grad.shape.get_volume() / length;

  return profile(backward_kernel,
                 profiling,
                 "[TopK] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 output_grad.get_float_ptr(),
                 indices.get_int32_ptr(),
                 input_grad.get_float_ptr(),
                 batch_size,
                 length,
                 attrs.k);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  TopKAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view) {
  auto env = sim_factory.new_environment();

  ParallelTensorShape output_shape = get_output_shape(attrs, input.shape);

  SimTaskBinding init_binding;
  init_binding.bind_arg(ATTRS, attrs);

  auto init_accessor = env.get_init_accessor(TOPK_INIT_TASK_ID, init_binding);
  DeviceSpecific<DeviceStates> per_device_state = init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;
  fwd_binding.bind_arg(PER_DEVICE_STATE, per_device_state);
  fwd_binding.bind_arg(PROFILING, profiling_settings());
  fwd_binding.bind(INPUT, input.shape);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind(INDICES, output_shape);
  fwd_binding.bind_arg(ATTRS, attrs);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor = env.get_fwd_accessor(TOPK_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(TOPK_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

TaskImplFunction get_topk_init_task_impl() {
  return init_task_impl;
}
TaskImplFunction get_topk_fwd_task_impl() {
  return forward_task_impl;
}
TaskImplFunction get_topk_bwd_task_impl() {
  return backward_task_impl;
}

OpTaskSignature get_topk_init_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_arg_slot<TopKAttrs>(ATTRS);
  init.add_return_value<TopKPerDeviceState>();

  return init;
}
OpTaskSignature get_topk_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_arg_slot<TopKAttrs>(ATTRS);
  fwd.add_unchecked_arg_slot<TopKPerDeviceState>(PER_DEVICE_STATE);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  fwd.add_output_slot(INDICES);
  return fwd;
}
OpTaskSignature get_topk_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_topk_fwd_signature());
  return bwd;
}

std::vector<task_id_t> get_task_ids(TopKAttrs const &) {
  return {TOPK_INIT_TASK_ID, TOPK_FWD_TASK_ID, TOPK_BWD_TASK_ID};
}

}; // namespace FlexFlow

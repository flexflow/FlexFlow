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

using namespace FlexFlow::Kernels::Reshape;

enum slots { INPUT, OUTPUT, ATTRS, PROFILING, PER_DEVICE_STATE };

OpTaskInvocation init(ReshapeAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(ATTRS, attrs);

  return {task_id_t::RESHAPE_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(ReshapeAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<ReshapePerDeviceState>());
  binding.bind_arg(PROFILING, profiling_settings());

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));
  return {task_id_t::RESHAPE_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(ReshapeAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {task_id_t::RESHAPE_BWD_TASK_ID, binding};
}

static DeviceSpecificDeviceStates
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto attrs = acc.get_argument<ReshapeAttrs>(ATTRS);

  ReshapePerDeviceState per_device_state = init_kernel(attrs.shape.data_type);
  return DeviceSpecificDeviceStates{
      DeviceSpecific<ReshapePerDeviceState>::create(per_device_state)};
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

TaskImplFunction get_reshape_init_task_impl() {
  return TaskImplFunction{InitTaskImplFunction{init_task_impl}};
}
TaskImplFunction get_reshape_fwd_task_impl() {
  return TaskImplFunction{FwdBwdTaskImplFunction{forward_task_impl}};
}
TaskImplFunction get_reshape_bwd_task_impl() {
  return TaskImplFunction{FwdBwdTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_reshape_init_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_arg_slot<ReshapeAttrs>(ATTRS);

  init.add_return_value<ReshapePerDeviceState>();
  return init;
}
OpTaskSignature get_reshape_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_unchecked_arg_slot<ReshapePerDeviceState>(PER_DEVICE_STATE);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  return fwd;
}
OpTaskSignature get_reshape_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_reshape_fwd_signature());
  return bwd;
}

std::vector<task_id_t> get_task_ids(ReshapeAttrs const &) {
  return {task_id_t::RESHAPE_INIT_TASK_ID,
          task_id_t::RESHAPE_FWD_TASK_ID,
          task_id_t::RESHAPE_BWD_TASK_ID};
}

}; // namespace FlexFlow

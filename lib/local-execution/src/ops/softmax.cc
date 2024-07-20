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

static DeviceSpecific<DeviceStates>
    init_task_impl(TaskArgumentAccessor const &acc) {
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);

  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto const &attrs = acc.get_argument<SoftmaxAttrs>(ATTRS);

  int output_w = output.shape.at(legion_dim_t(0));
  int output_h = output.shape.at(legion_dim_t(1));
  int output_c = output.shape.at(legion_dim_t(2));
  int output_n = output.shape.at(legion_dim_t(3));

  SoftmaxPerDeviceState per_device_state = init_kernel(
      handle, attrs.dim.value, output_n, output_c, output_h, output_w);

  return DeviceSpecific<DeviceStates>::create(per_device_state);
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto per_device_state =
      acc.get_argument<SoftmaxPerDeviceState>(PER_DEVICE_STATE);

  return profile(forward_kernel,
                 profiling,
                 "[SoftMax] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr());
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  assert(input_grad.shape == input.shape);

  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);
  auto output = acc.get_tensor<Permissions::RO>(OUTPUT);

  assert(output_grad.shape == output.shape);

  return profile(backward_kernel,
                 profiling,
                 "[SoftMax] backward_time = {:.2lf}ms\n",
                 input_grad.get_float_ptr(),
                 output_grad.get_float_ptr(),
                 output_grad.shape.get_volume());
}

TaskImplFunction get_softmax_init_task_impl() {
  return TaskImplFunction{init_task_impl};
}
TaskImplFunction get_softmax_fwd_task_impl() {
  return TaskImplFunction{forward_task_impl};
}
TaskImplFunction get_softmax_bwd_task_impl() {
  return TaskImplFunction{backward_task_impl};
}

OpTaskSignature get_softmax_init_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);
  init.add_arg_slot<SoftmaxAttrs>(ATTRS);
  init.add_return_value<SoftmaxPerDeviceState>();
  return init;
}
OpTaskSignature get_softmax_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_unchecked_arg_slot<SoftmaxPerDeviceState>(PER_DEVICE_STATE);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  return fwd;
}
OpTaskSignature get_softmax_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_softmax_fwd_signature());
  return bwd;
}

std::vector<task_id_t> get_task_ids(SoftmaxAttrs const &) {
  return {SOFTMAX_INIT_TASK_ID, SOFTMAX_FWD_TASK_ID, SOFTMAX_BWD_TASK_ID};
}

}; // namespace FlexFlow

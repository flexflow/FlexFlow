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

#include "gather.h"
#include "kernels/gather_kernels.h"
#include "local-execution/legion_tensor_shape.h"
#include "op-attrs/get_output_shapes.h"
#include <optional>

namespace FlexFlow {

using namespace FlexFlow::Kernels::Gather;

enum Slots { INPUT, OUTPUT, INDEX, ATTRS, HANDLE, PROFILING, PER_DEVICE_STATE };

OpTaskInvocation init(GatherAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(INPUT, input_tensor(0));
  binding.bind(INDEX, input_tensor(1));
  binding.bind(OUTPUT, output_tensor(0));
  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(HANDLE, ff_handle());

  return {GATHER_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(GatherAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<GatherPerDeviceState>());

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));
  binding.bind(INDEX, weight_tensor(0));

  return {GATHER_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(GatherAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {GATHER_BWD_TASK_ID, binding};
}

static DeviceSpecific<DeviceStates>
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto index = acc.get_tensor<Permissions::RO>(INDEX);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);
  auto const &attrs = acc.get_argument<GatherAttrs>(ATTRS);
  legion_dim_t legion_dim =
      legion_dim_from_ff_dim(attrs.dim, input.shape.num_dims());

  assert(input.shape.get_dim() == index.shape.get_dim());
  assert(output.shape.get_dim() == index.shape.get_dim());

  for (int i = 0; i < input.shape.get_dim(); i++) {
    assert(index.shape[legion_dim_t(i)] == output.shape[legion_dim_t(i)]);
    if (i != legion_dim.value) {
      assert(input.shape[legion_dim_t(i)] == index.shape[legion_dim_t(i)]);
    }
  }

  GatherPerDeviceState per_device_state = {handle, legion_dim};
  return DeviceSpecific<DeviceStates>::create(per_device_state);
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto per_device_state =
      acc.get_argument<GatherPerDeviceState>(PER_DEVICE_STATE);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto index = acc.get_tensor<Permissions::RO>(INDEX);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 "[Gather] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input,
                 index,
                 output);
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto per_device_state =
      acc.get_argument<GatherPerDeviceState>(PER_DEVICE_STATE);

  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);
  auto index = acc.get_tensor<Permissions::RO>(INDEX);
  auto input_grad = acc.get_tensor_grad<Permissions::WO>(INPUT);

  return profile(backward_kernel,
                 profiling,
                 "[Gather] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 output_grad,
                 index,
                 input_grad);
}

TaskImplFunction get_gather_init_task_impl() {
  return TaskImplFunction{init_task_impl};
}
TaskImplFunction get_gather_fwd_task_impl() {
  return TaskImplFunction{forward_task_impl};
}
TaskImplFunction get_gather_bwd_task_impl() {
  return TaskImplFunction{backward_task_impl};
}

OpTaskSignature get_gather_init_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_input_slot(INPUT);
  init.add_input_slot(INDEX);
  init.add_output_slot(OUTPUT);

  init.add_arg_slot<GatherAttrs>(ATTRS);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  init.add_return_value<GatherPerDeviceState>();

  return init;
}

OpTaskSignature get_gather_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<bool>(PROFILING);
  fwd.add_arg_slot<GatherAttrs>(ATTRS);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  fwd.add_weight_slot(INDEX);

  return fwd;
}

OpTaskSignature get_gather_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_gather_fwd_signature());

  return bwd;
}

std::vector<task_id_t> get_task_ids(GatherAttrs const &) {
  return {GATHER_INIT_TASK_ID, GATHER_FWD_TASK_ID, GATHER_BWD_TASK_ID};
}

}; // namespace FlexFlow

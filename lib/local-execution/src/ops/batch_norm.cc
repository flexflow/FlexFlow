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

#include "batch_norm.h"
#include "kernels/batch_norm_kernels.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::BatchNorm;

enum Slots {
  INPUT,  // tensor
  SCALE,  // tensor
  BIAS,   // tensor
  OUTPUT, // tensor
  ATTRS,
  PROFILING,
  PER_DEVICE_STATE,
  RELU,
  HANDLE
};

OpTaskInvocation init(BatchNormAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(INPUT, input_tensor(0));
  binding.bind(BIAS, input_tensor(2));
  binding.bind(OUTPUT, output_tensor(0));

  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(HANDLE, ff_handle());

  return {task_id_t::BATCHNORM_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(BatchNormAttrs const &attrs) {
  OpTaskBinding binding;
  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<BatchNormPerDeviceState>());

  binding.bind(INPUT, input_tensor(0));
  binding.bind(SCALE, input_tensor(1));
  binding.bind(BIAS, input_tensor(2));
  binding.bind(OUTPUT, output_tensor(0));

  return {task_id_t::BATCHNORM_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(BatchNormAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {task_id_t::BATCHNORM_BWD_TASK_ID, binding};
}

static DeviceSpecificDeviceStates
    init_task_impl(TaskArgumentAccessor const &acc) {
  Allocator allocator = acc.get_allocator();
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto const &attrs = acc.get_argument<BatchNormAttrs>(ATTRS);

  int output_w = output.shape[legion_dim_t(0)];
  int output_h = output.shape[legion_dim_t(1)];
  int output_c = output.shape[legion_dim_t(2)];
  int output_n = output.shape[legion_dim_t(3)];

  float *runningMean;

  BatchNormPerDeviceState per_device_state = init_kernel(handle,
                                                         allocator,
                                                         runningMean,
                                                         output_n,
                                                         output_c,
                                                         output_h,
                                                         output_w,
                                                         attrs.relu);

  return DeviceSpecificDeviceStates{
      DeviceSpecific<BatchNormPerDeviceState>::create(per_device_state)};
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_argument<BatchNormPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto scale = acc.get_tensor<Permissions::RO>(SCALE);
  auto bias = acc.get_tensor<Permissions::RO>(SCALE);

  return profile(forward_kernel,
                 profiling,
                 "[BatchNorm] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr(),
                 scale.get_float_ptr(),
                 bias.get_float_ptr());
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_argument<BatchNormPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto output = acc.get_tensor<Permissions::RO>(OUTPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RW>(OUTPUT);
  auto scale = acc.get_tensor<Permissions::RO>(SCALE);
  auto scale_grad = acc.get_tensor_grad<Permissions::RW>(SCALE);
  auto bias_grad = acc.get_tensor_grad<Permissions::RW>(BIAS);

  return profile(backward_kernel,
                 profiling,
                 "[BatchNorm] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output_grad.get_float_ptr(),
                 output.get_float_ptr(),
                 input_grad.get_float_ptr(),
                 scale.get_float_ptr(),
                 scale_grad.get_float_ptr(),
                 bias_grad.get_float_ptr(),
                 output.shape.get_volume());
}

TaskImplFunction get_batch_norm_init_task_impl() {
  return TaskImplFunction{InitTaskImplFunction{init_task_impl}};
}
TaskImplFunction get_batch_norm_fwd_task_impl() {
  return TaskImplFunction{FwdBwdTaskImplFunction{forward_task_impl}};
}
TaskImplFunction get_batch_norm_bwd_task_impl() {
  return TaskImplFunction{FwdBwdTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_batch_norm_init_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_input_slot(INPUT);
  init.add_input_slot(BIAS);
  init.add_output_slot(OUTPUT);
  init.add_arg_slot<BatchNormAttrs>(ATTRS);
  init.add_arg_slot<bool>(PROFILING);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  return init;
}

OpTaskSignature get_batch_norm_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(INPUT);
  fwd.add_input_slot(SCALE);
  fwd.add_input_slot(BIAS);
  fwd.add_output_slot(OUTPUT);
  fwd.add_arg_slot<bool>(PROFILING);
  fwd.add_unchecked_arg_slot<BatchNormPerDeviceState>(PER_DEVICE_STATE);

  return fwd;
}
OpTaskSignature get_batch_norm_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_batch_norm_fwd_signature());

  return bwd;
}

std::vector<task_id_t> get_task_ids(BatchNormAttrs const &) {
  return {
      task_id_t::BATCHNORM_INIT_TASK_ID,
      task_id_t::BATCHNORM_FWD_TASK_ID,
      task_id_t::BATCHNORM_BWD_TASK_ID,
  };
}

}; // namespace FlexFlow

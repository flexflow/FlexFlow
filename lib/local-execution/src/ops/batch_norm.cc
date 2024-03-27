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

  return {BATCHNORM_INIT_TASK_ID, binding};
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

  return {BATCHNORM_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(BatchNormAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {BATCHNORM_BWD_TASK_ID, binding};
}

static DeviceSpecific<BatchNormPerDeviceState>
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

  DeviceSpecific<BatchNormPerDeviceState> per_device_state =
          init_kernel(handle,
                      allocator,
                      runningMean,
                      output_n,
                      output_c,
                      output_h,
                      output_w,
                      attrs.relu);

  return per_device_state;
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
                 "[BatchNorm] forward_time = %.2lfms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr(),
                 scale.get_float_ptr(),
                 bias.get_float_ptr());
}



static std::optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
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
                 "[BatchNorm] backward_time = %.2lfms\n",
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



CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  BatchNormAttrs const &attrs,
                                  InputParallelTensorDesc const &input_shape,
                                  InputParallelTensorDesc const &scale_shape,
                                  InputParallelTensorDesc const &bias_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv) {
  auto env = sim.new_environment();

  ParallelTensorShape output_shape = get_output_shape(attrs);

  SimTaskBinding init_binding;
  init_binding.bind(INPUT, input_shape);
  init_binding.bind(BIAS, bias_shape);
  init_binding.bind(OUTPUT, output_shape);

  init_binding.bind_arg(ATTRS, attrs);
  init_binding.bind_arg(PROFILING, settings);
  init_binding.bind_arg(HANDLE, ff_handle());

  auto init_accessor =
      env.get_init_accessor(ATTENTION_INIT_TASK_ID, init_binding);
  DeviceSpecific<BatchNormPerDeviceState> per_device_state =
      init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;
  fwd_binding.bind(INPUT, input_shape);
  fwd_binding.bind(SCALE, scale_shape);
  fwd_binding.bind(BIAS, bias_shape);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind_arg(PER_DEVICE_STATE, per_device_state);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor = env.get_fwd_accessor(ATTENTION_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(ATTENTION_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
OpTaskSignature init_signature<BATCHNORM_INIT_TASK_ID>() {
  OpTaskSignature init; init.type = OpTaskType::INIT;
  init.add_input_slot(INPUT);
  init.add_input_slot(BIAS);
  init.add_output_slot(OUTPUT);
  init.add_arg_slot<BatchNormAttrs>(ATTRS);
  init.add_arg_slot<bool>(PROFILING);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  return init;
}

template <>
void register_task<BATCHNORM_INIT_TASK_ID>() {
  register_task(BATCHNORM_INIT_TASK_ID,
                "BatchNorm Init",
                init_signature<BATCHNORM_INIT_TASK_ID>(),
                init_task_impl);
}

template <>
OpTaskSignature fwd_signature<BATCHNORM_FWD_TASK_ID>() {
  OpTaskSignature fwd; fwd.type = OpTaskType::FWD;

  fwd.add_input_slot(INPUT);
  fwd.add_input_slot(SCALE);
  fwd.add_input_slot(BIAS);
  fwd.add_output_slot(OUTPUT);
  fwd.add_arg_slot<bool>(PROFILING);
  fwd.add_unchecked_arg_slot<BatchNormPerDeviceState>(PER_DEVICE_STATE);

  return fwd;
}

template <>
void register_task<BATCHNORM_FWD_TASK_ID>() {
  register_task(BATCHNORM_FWD_TASK_ID,
                "BatchNorm Fwd",
                fwd_signature<BATCHNORM_FWD_TASK_ID>(),
                forward_task_impl);
}

template <>
OpTaskSignature bwd_signature<BATCHNORM_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(fwd_signature<BATCHNORM_FWD_TASK_ID>());

  return bwd;
}

template <>
void register_task<BATCHNORM_BWD_TASK_ID>() {
  register_task(BATCHNORM_BWD_TASK_ID,
                "BatchNorm Bwd",
                bwd_signature<BATCHNORM_BWD_TASK_ID>(),
                backward_task_impl);
}

}; // namespace FlexFlow

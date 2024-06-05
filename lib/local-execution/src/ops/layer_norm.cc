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

#include "layer_norm.h"
#include "kernels/layer_norm_kernels.h"
#include "op-attrs/get_output_shapes.h"
#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"
#include <type_traits>

namespace FlexFlow {

using namespace FlexFlow::Kernels::LayerNorm;

enum Slots {
  PROFILING,
  INPUT,
  OUTPUT,
  GAMMA,
  BETA,
  PER_DEVICE_STATE,
  ATTRS,
  HANDLE
};

OpTaskInvocation init(LayerNormAttrs const &attrs) {
  OpTaskBinding b;

  b.bind(INPUT, input_tensor(0));

  b.bind_arg(HANDLE, ff_handle());
  b.bind_arg(ATTRS, attrs);

  return {LAYERNORM_INIT_TASK_ID, b};
}

OpTaskInvocation forward(LayerNormAttrs const &attrs) {
  OpTaskBinding b;

  b.bind(INPUT, input_tensor(0));
  b.bind(OUTPUT, output_tensor(0));
  b.bind(GAMMA, weight_tensor(0)); // todo, this may have some problem
  b.bind(BETA, weight_tensor(1));  // how to get gmmam and beta
  b.bind_arg(PROFILING, profiling_settings());
  b.bind_arg(PER_DEVICE_STATE, per_device_op_state<LayerNormPerDeviceState>());

  return {LAYERNORM_FWD_TASK_ID, b};
}

OpTaskInvocation backward(LayerNormAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {LAYERNORM_BWD_TASK_ID, b};
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto gamma = acc.get_tensor<Permissions::RW>(GAMMA);
  auto beta = acc.get_tensor<Permissions::RW>(BETA);

  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto &state = acc.get_argument<LayerNormPerDeviceState>(PER_DEVICE_STATE);

  return profile(forward_kernel,
                 profiling,
                 "[LayerNorm] forward time = {:.2lf}ms\n",
                 state,
                 input,
                 output,
                 gamma,
                 beta);
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto gamma = acc.get_tensor<Permissions::RO>(GAMMA);

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto gamma_grad = acc.get_tensor_grad<Permissions::RW>(GAMMA);
  auto beta_grad = acc.get_tensor_grad<Permissions::RW>(BETA);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);

  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto &state = acc.get_argument<LayerNormPerDeviceState>(PER_DEVICE_STATE);

  return profile(backward_kernel,
                 profiling,
                 "[LayerNorm] backward time = {:.2lf}ms\n",
                 state,
                 output_grad,
                 input,
                 input_grad,
                 gamma,
                 gamma_grad,
                 beta_grad);
}

static DeviceSpecific<LayerNormPerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto const &attrs = acc.get_argument<LayerNormAttrs>(ATTRS);
  Allocator allocator = acc.get_allocator();
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);

  // question: how to get batch_size and effective_num_elements
  int64_t effective_batch_size, effective_num_elements;
  int M = 1;
  for (int i = 0; i < attrs.axes.size(); i++) {
    M *= input.shape.at(legion_dim_t(attrs.axes[i]));
  }
  int num_replicas = 1;
  for (int i = 0; i < input.shape.num_dims(); i++) {
    num_replicas *= input.shape.at(legion_dim_t(i));
    effective_num_elements = M;
    effective_batch_size = input.shape.get_volume() / M;
  }

  DeviceSpecific<LayerNormPerDeviceState> per_device_state =
      init_kernel(handle,
                  allocator,
                  attrs.elementwise_affine,
                  effective_batch_size,
                  effective_num_elements,
                  attrs.eps);
  return per_device_state;
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  LayerNormAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view) {
  auto env = sim_factory.new_environment();
  ParallelTensorShape output_shape = get_output_shape(attrs, input.shape);

  SimTaskBinding init_binding;
  init_binding.bind_arg(HANDLE, ff_handle());
  init_binding.bind_arg(ATTRS, attrs);
  init_binding.bind(INPUT, input.shape);

  auto init_accessor =
      env.get_init_accessor(LAYERNORM_INIT_TASK_ID, init_binding);

  DeviceSpecific<LayerNormPerDeviceState> per_device_state =
      init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;
  fwd_binding.bind(INPUT, input.shape);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind_arg(PER_DEVICE_STATE, per_device_state);

  fwd_binding.bind(GAMMA, input.shape);
  fwd_binding.bind(BETA, input.shape);
  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor = env.get_fwd_accessor(LAYERNORM_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(LAYERNORM_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
OpTaskSignature fwd_signature<LAYERNORM_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  fwd.add_weight_slot(GAMMA);
  fwd.add_weight_slot(BETA);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_unchecked_arg_slot<LayerNormPerDeviceState>(PER_DEVICE_STATE);
  return fwd;
}

template <>
OpTaskSignature bwd_signature<LAYERNORM_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(fwd_signature<LAYERNORM_FWD_TASK_ID>());
  return bwd;
}

template <>
OpTaskSignature init_signature<LAYERNORM_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_input_slot(INPUT);
  init.add_arg_slot<LayerNormAttrs>(ATTRS);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  init.add_return_value<LayerNormPerDeviceState>();
  return init;
}

template <>
void register_task<LAYERNORM_INIT_TASK_ID>() {

  register_task(LAYERNORM_INIT_TASK_ID,
                "LayerNorm init",
                init_signature<LAYERNORM_INIT_TASK_ID>(),
                init_task_impl);
}

template <>
void register_task<LAYERNORM_FWD_TASK_ID>() {
  register_task(LAYERNORM_FWD_TASK_ID,
                "LayerNorm forward",
                fwd_signature<LAYERNORM_FWD_TASK_ID>(),
                forward_task_impl);
}

template <>
void register_task<LAYERNORM_BWD_TASK_ID>() {
  register_task(LAYERNORM_BWD_TASK_ID,
                "LayerNorm backward",
                bwd_signature<LAYERNORM_BWD_TASK_ID>(),
                backward_task_impl);
}

} // namespace FlexFlow

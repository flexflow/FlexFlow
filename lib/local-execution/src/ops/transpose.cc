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

#include "transpose.h"
#include "kernels/transpose_kernels.h"
#include "op-attrs/get_output_shapes.h"
#include "op-attrs/ops/transpose.h"
#include "utils/exception.decl.h"

using namespace FlexFlow::Kernels::Transpose;

namespace FlexFlow {

enum Slots {
  INPUT,  // tensor
  OUTPUT, // tensor
  ATTRS,
  PROFILING,
  PER_DEVICE_STATE,
};

OpTaskInvocation init(TransposeAttrs const &attrs) {
  OpTaskBinding binding;
  binding.bind_arg(ATTRS, attrs);
  return {TRANSPOSE_INIT_TASK_ID, binding};
}

static DeviceSpecific<TransposePerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto const &attrs = acc.get_argument<TransposeAttrs>(ATTRS);
  std::vector<ff_dim_t> perm = static_cast<std::vector<ff_dim_t>>(attrs.perm);
  DeviceSpecific<TransposePerDeviceState> per_device_state =
      init_kernel(perm.size(), perm);

  return per_device_state;
}

// TODO: OpTaskSignature

// template <>
// void register_task<TRANSPOSE_INIT_TASK_ID>() {
// OpTaskSignature init(OpTaskType::INIT);

//     init.add_arg_slot<TransposeAttrs>(ATTRS);

// init.add_return_value<TransposePerDeviceState>();

// register_task(TRANSPOSE_INIT_TASK_ID, "Transpose::init", init,
// init_task_impl);
// }

OpTaskInvocation forward(TransposeAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<TransposePerDeviceState>());
  binding.bind_arg(PROFILING, profiling_settings());

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {TRANSPOSE_FWD_TASK_ID, binding};
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto per_device_state =
      acc.get_argument<TransposePerDeviceState>(PER_DEVICE_STATE);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 "[Transpose] Forward_time = {:.2lf} [ms]",
                 per_device_state,
                 input,
                 output);
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto per_device_state =
      acc.get_argument<TransposePerDeviceState>(PER_DEVICE_STATE);

  auto input_grad = acc.get_tensor_grad<Permissions::WO>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 "[Transpose] Backward_time = {:.2lf} [ms]",
                 per_device_state,
                 input_grad,
                 output_grad);
}

OpTaskInvocation backward(TransposeAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {TRANSPOSE_BWD_TASK_ID, binding};
}

CostMetrics
    measure_operator_cost(SimEnvFactory const &sim_factory,
                          TransposeAttrs const &attrs,
                          InputVariadicParallelTensorDesc const
                              &input_descs, // Note:this may have some problem
                          ProfilingSettings const &settings,
                          MachineView const &machine_view) {
  auto env = sim_factory.new_environment();

  SimTaskBinding init_binding;
  init_binding.bind_arg(ATTRS, attrs);

  auto init_accessor =
      env.get_init_accessor(TRANSPOSE_INIT_TASK_ID, init_binding);
  DeviceSpecific<TransposePerDeviceState> per_device_state =
      init_task_impl(init_accessor);

  ParallelTensorShape output_shape =
      get_output_shape(attrs, input_descs.shapes);

  SimTaskBinding fwd_binding;
  fwd_binding.bind_arg(PER_DEVICE_STATE, per_device_state);
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind(INPUT, input_descs.shapes);
  fwd_binding.bind(OUTPUT, output_shape);

  auto fwd_accessor = env.get_fwd_accessor(TRANSPOSE_FWD_TASK_ID, fwd_binding);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(TRANSPOSE_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

} // namespace FlexFlow

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

#include "op-attrs/get_output_shapes.h"
#include "op_task_signature.h"
#include "utils/hash-utils.h"
#include "variadic_tensor_ref.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Concat;

enum Slots { INPUTS, OUTPUT, ATTRS, PROFILING, HANDLE, NUM_INPUTS };

OpTaskInvocation forward(ConcatAttrs const &attrs) {
  OpTaskBinding binding;
  binding.bind(INPUTS, get_input_tensors());
  binding.bind(OUTPUT, output_tensor(0));
  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(ATTRS, attrs);

  return {CONCAT_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(ConcatAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {CONCAT_BWD_TASK_ID, b};
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto const &attrs = acc.get_argument<ConcatAttrs>(ATTRS);

  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto inputs = acc.get_variadic_tensor<Permissions::RO>(INPUTS);

  assert(attrs.num_inputs <= MAX_NUM_INPUTS);

  return profile(forward_kernel,
                 profiling,
                 "[Concat] forward_time = %.2lfms\n",
                 output,
                 inputs,
                 attrs.axis);
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto const &attrs = acc.get_argument<ConcatAttrs>(ATTRS);

  auto input_grads = acc.get_variadic_tensor_grad<Permissions::RW>(INPUTS);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);

  assert(attrs.num_inputs <= MAX_NUM_INPUTS);

  return profile(backward_kernel,
                 profiling,
                 "[Concat] backward_time = %.2lfms\n",
                 output_grad,
                 input_grads,
                 attrs.axis);
}

CostMetrics
    measure_operator_cost(SimEnvFactory const &sim,
                          ConcatAttrs const &attrs,
                          InputVariadicParallelTensorDesc const &inputs_shape,
                          ProfilingSettings const &settings,
                          MachineView const &mv) {
  assert(attrs.num_inputs <= MAX_NUM_INPUTS);

  auto env = sim.new_environment();

  ParallelTensorShape output_shape =
      get_output_shape(attrs, inputs_shape.shapes);

  SimTaskBinding fwd_binding;
  fwd_binding.bind_arg(ATTRS, attrs);
  fwd_binding.bind(INPUTS, inputs_shape);
  fwd_binding.bind(OUTPUT, output_shape);
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
OpTaskSignature fwd_signature<CONCAT_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<ConcatAttrs>(ATTRS);
  fwd.add_arg_slot<bool>(PROFILING);
  fwd.add_input_slot(INPUTS, SlotType::VARIADIC);
  fwd.add_output_slot(OUTPUT);

  return fwd;
}

template <>
void register_task<CONCAT_FWD_TASK_ID>() {
  register_task(CONCAT_FWD_TASK_ID,
                "Concat Fwd",
                fwd_signature<CONCAT_FWD_TASK_ID>(),
                forward_task_impl);
}

template <>
OpTaskSignature bwd_signature<CONCAT_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(fwd_signature<CONCAT_FWD_TASK_ID>());

  return bwd;
}

template <>
void register_task<CONCAT_BWD_TASK_ID>() {
  register_task(CONCAT_BWD_TASK_ID,
                "Concat Bwd",
                bwd_signature<CONCAT_BWD_TASK_ID>(),
                backward_task_impl);
}

}; // namespace FlexFlow

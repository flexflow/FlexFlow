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

#include "replicate.h"
#include "kernels/replicate_kernels.h"
#include "op-attrs/get_output_shapes.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/exception.h"
#include "utils/graph/serialparallel.h"
#include "utils/hash-utils.h"
#include <sys/types.h>

namespace FlexFlow {
// declare Legion names



using namespace FlexFlow::Kernels::Replicate;

enum Slots { INPUT, OUTPUT, ATTRS, PROFILING };

OpTaskInvocation forward(ReplicateAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PROFILING, profiling_settings());

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));
  binding.bind_arg(ATTRS, attrs);

  return {REPLICATE_FWD_TASK_ID, binding};
}
OpTaskInvocation backward(ReplicateAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {REPLICATE_BWD_TASK_ID, binding};
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 "[replicate] forward_time = %.2lfms\n",
                 input,
                 output);
}



static std::optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input_grad = acc.get_tensor_grad<Permissions::WO>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);
  auto const & attrs = acc.get_argument<ReplicateAttrs>(ATTRS);

  return profile(backward_kernel,
                 profiling,
                 "[replicate] backward_time = %.2lfms\n",
                 input_grad,
                 output_grad,
                 attrs.replicate_degree); // is this `num_replicas`?
}



CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ReplicateAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view) {
  auto env = sim_factory.new_environment();
  SimTaskBinding fwd_binding;
  fwd_binding.bind_arg(PROFILING, settings);
  ParallelTensorShape output = get_output_shape(attrs, input.shape);
  fwd_binding.bind(INPUT, input.shape);
  fwd_binding.bind(OUTPUT, output);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);
  auto fwd_accessor = env.get_fwd_accessor(REPLICATE_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(REPLICATE_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<REPLICATE_FWD_TASK_ID>() {
  OpTaskSignature fwd; fwd.type = OpTaskType::FWD;

  fwd.add_arg_slot<bool>(PROFILING);
  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  register_task(REPLICATE_FWD_TASK_ID, "Replicate fwd", fwd, forward_task_impl);
}

// TODO: OpTaskSignature


// template <>
// void register_task<REPLICATE_BWD_TASK_ID>() {
//   OpTaskSignature bwd = infer_bwd_signature(get_op_signature(CAST_FWD_TASK_ID));

//   register_task(REPLICATE_BWD_TASK_ID, "Replicate bwd", bwd, backward_task_impl);
// }

}; // namespace FlexFlow

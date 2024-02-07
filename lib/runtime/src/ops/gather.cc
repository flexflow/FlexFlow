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
#include "op-attrs/get_output_shapes.h"

namespace FlexFlow {

// declare Legion names
using Legion::Context;
using Legion::PhysicalRegion;
using Legion::Runtime;
using Legion::Task;

using namespace FlexFlow::Kernels::Gather;

enum Slots { INPUT, OUTPUT, INDEX, ATTRS, PROFILING };

OpTaskInvocation forward(GatherAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(PROFILING, profiling_settings());

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));
  binding.bind(INDEX, weight_tensor(0));

  return {GATHER_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(GatherAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {GATHER_BWD_TASK_ID, binding};
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto attrs = acc.get_argument<GatherAttrs>(ATTRS);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto index = acc.get_tensor<Permissions::RO>(INDEX);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 "[Gather] forward_time = %.2lfms\n",
                 input,
                 index,
                 output,
                 attrs.stride,
                 attrs.dim);
}

static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  forward_task_impl(acc);
}

static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto attrs = acc.get_argument<GatherAttrs>(ATTRS);

  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);
  auto index = acc.get_tensor<Permissions::RO>(INDEX);
  auto input_grad = acc.get_tensor_grad<Permissions::WO>(INPUT);

  return profile(forward_kernel,
                 profiling,
                 "[Gather] forward_time = %.2lfms\n",
                 output_grad,
                 index,
                 input_grad,
                 attrs.stride,
                 attrs.dim);
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  GatherAttrs const &attrs,
                                  InputParallelTensorDesc const &input_shape,
                                  InputParallelTensorDesc const &index_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv) {

  auto env = sim.new_environment();

  std::vector<ParallelTensorShape> output_shape =
      get_output_shapes(attrs, input_shape.shape, index_shape.shape);

  SimTaskBinding fwd_binding;
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind_arg(ATTRS, attrs);

  fwd_binding.bind(INPUT, input_shape);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind(INDEX, index_shape);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor = env.get_fwd_accessor(GATHER_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(GATHER_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <task_id_t>
OpTaskSignature fwd_signature<GATHER_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<bool>(PROFILING);
  fwd.add_arg_slot<GatherAttrs>(ATTRS);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  fwd.add_weight_slot(INDEX);

  return fwd;
}

template <>
void register_task<GATHER_FWD_TASK_ID>() {
  register_task(GATHER_FWD_TASK_ID,
                "Gather Fwd",
                fwd_signature<GATHER_FWD_TASK_ID>(),
                forward_task);
}

template <task_id_t>
OpTaskSignature bwd_signature<GATHER_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(fwd_signature<GATHER_FWD_TASK_ID>());

  return bwd;
}

template <>
void register_task<GATHER_BWD_TASK_ID>() {
  register_task(GATHER_BWD_TASK_ID,
                "Gather Bwd",
                bwd_signature<GATHER_BWD_TASK_ID>(),
                backward_task);
}

}; // namespace FlexFlow

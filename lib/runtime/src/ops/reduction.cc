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

#include "parallel_ops/reduction.h"
#include "kernels/reduction_kernels.h"
#include "op-attrs/get_output_shape.h"
#include "utils/exception.decl.h"
#include "utils/hash-utils.h"

namespace FlexFlow {
// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::LogicalPartition;
using Legion::LogicalRegion;
using Legion::Machine;
using Legion::Memory;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

using namespace FlexFlow::Kernels::Reduction;

ReductionParams Reduction::get_params() const {
  ReductionParams params;
  params.reduction_legion_dim = this->reduction_dim;
  params.reduction_degree = this->reduction_degree;
  return params;
}

enum Slots { INPUT, OUTPUT, ATTRS, PROFILING };

OpTaskInvocation forward(ReductionAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(ATTRS, attrs);
  binding.bind(INPUT, input_parallel_tensor_shape(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {REDUCTION_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(ReductionAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {REDUCTION_BWD_TASK_ID, binding};
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling_settings =
      acc.get_argument<ProfilingSettings>(PROFILING);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto attrs = acc.get_argument<ReductionAttrs>(ATTRS);
  // Note: how to get num_replicas?
  size_t num_replicas = attrs.reduction_dim;

  return profiling(forward_kernel,
                   profiling_settings,
                   "[Reduction] forward_time = %.2lfms\n",
                   input,
                   output,
                   num_replicas);
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

  auto input_grad = acc.get_tensor_grad<Permissions::RO>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::WO>(OUTPUT);
  return profiling(backward_kernel,
                   profiling,
                   "[Reduction] backward_time = %.2lfms\n",
                   input_grad,
                   output_grad);
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ReductionAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view) {
  ParallelTensorShape output_shape = get_output_shape(input, attrs);

  auto env = sim.new_environment();

  SimTaskBinding fwd_binding;
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind_arg(ATTRS, attrs);
  fwd_binding.bind(INPUT, input);
  fwd.binding.bind(OUTPUT, output_shape);

  auto fwd_accessor = env.get_fwd_accessor(REDUCTION_FWD_TASK_ID, fwd_binding);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(REDUCTION_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);

  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<REDUCTION_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  register_task(REDUCTION_FWD_TASK_ID, "Reduction Fwd", fwd, forward_task);
}

template <>
void register_task<REDUCTION_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(get_op_signature(REDUCTION_FWD_TASK_ID));

  register_task(REDUCTION_BWD_TASK_ID, "Reduction Bwd", bwd, backward_task);
}

}; // namespace FlexFlow

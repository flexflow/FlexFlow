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

#include "parallel_ops/replicate.h"
#include "kernels/replicate_kernels.h"
#include "utils/exception.decl.h"
#include "utils/hash-utils.h"
#include <sys/types.h>

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

using namespace FlexFlow::Kernels::Replicate;

enum Slots { INPUT, OUTPUT, ATTRS, PROFILING };

/* Params */
bool operator==(ReplicateParams const &lhs, ReplicateParams const &rhs) {
  return lhs.replicate_legion_dim == rhs.replicate_legion_dim &&
         lhs.replicate_degree == rhs.replicate_degree;
}

bool ReplicateParams::is_valid(ParallelTensorShape const &input) const {
  return input.is_valid();
}

ReplicateParams Replicate::get_params() const {
  ReplicateParams params;
  params.replicate_legion_dim = this->replicate_dim;
  params.replicate_degree = this->replicate_degree;
  return params;
}

OpTaskInvocation init(ReplicateAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(INPUT, input_parallel_tensor_shape(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {REPLICATE_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(ReplicateAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PROFILING, profiling_settings());

  binding.bind(INPUT, input_parallel_tensor_shape(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {REPLICATE_FWD_TASK_ID, binding};
}
OpTaskInvocation backward(ReplicateAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {REPLICATE_BWD_TASK_ID, binding};
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 "[replicate] forward_time = %.2lfms\n",
                 input,
                 output);
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

  return profile(backward_kernel,
                 profiling,
                 "[replicate] backward_time = %.2lfms\n",
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
                                  ReplicateAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view) {
  // Note(lambda): Does replicate has cost? currently I assume the replicate has
  // no cost
  auto env = sim.new_environment();
  SimTaskBinding fwd_binding;
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind(INPUT, input_parallel_tensor_shape(0));
  fwd_binding.bind(OUTPUT, output_tensor(0));

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);
  auto fwd_accessor = env.get_fwd_accessor(TOPK_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(TOPK_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<REPLICATE_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_input_slot(INPUT);
  init.add_output_slot(OUTPUT);

  // TODO: should we implement the init_task? how to do it?
  // register_task(REPLICATE_INIT_TASK_ID, "Replicate init", init , init_task);
}

template <>
void register_task<REPLICATE_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<bool>(PROFILING);
  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  register_task(REPLICATE_FWD_TASK_ID, "Replicate fwd", fwd, forward_task);
}

template <>
void register_task<REPLICATE_BWD_TASK_ID>() {
  OpTaskSignature bwd = infer_bwd_signature(get_op_signature(CAST_FWD_TASK_ID));

  register_task(REPLICATE_BWD_TASK_ID, "Replicate bwd", bwd, backward_task);
}

}; // namespace FlexFlow

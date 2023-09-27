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

#include "split.h"
#include "kernels/split_kernels.h"
#include "utils/exceptions.h"
#include "utils/hash-utils.h"

namespace FlexFlow {
// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;
using PCG::Node;

using namespace FlexFlow::Kernels::Split;

bool operator==(SplitParams const &lhs, SplitParams const &rhs) {
  return lhs.splits == rhs.splits && lhs.legion_axis == rhs.legion_axis;
}

bool SplitParams::is_valid(ParallelTensorShape const &input) const {
  return input.is_valid();
}

SplitParams Split::get_params() const {
  SplitParams params;
  params.splits = this->splits;
  params.legion_axis = this->legion_axis;
  return params;
}

enum Slots { INPUT, OUTPUT, ATTRS, PROFILING };

OpTaskInvocation init(SplitAttrs const &attr) {
  OpTaskBinding binding;

  binding.bind(INPUT, input_parallel_tensor_shape(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {SPLIT_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(SplitAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind(INPUT, input_parallel_tensor_shape(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {SPLIT_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(SplitAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {SPLIT_BWD_TASK_ID, binding};
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  acc.get_argument<CastPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  // Note: forward_kernel needs parameter Legion::coord_t const
  // *out_blk_sizes,Legion::coord_t in_blk_size, Legion::coord_t num_blks, int
  // numOutputs how to get these parameter?
  NOT_IMPLEMENTED();
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

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::WO>(OUTPUT);
  // Note: backward_kernel needs parameter Legion::coord_t const
  // *out_blk_sizes,Legion::coord_t in_blk_size, Legion::coord_t num_blks, int
  // numOutputs how to get these parameter?
  NOT_IMPLEMENTED();
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  SplitAttrs const &attrs,
                                  InputParallelTensorDes const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view) {
  auto env = sim.new_environment();

  ParallelTensorShape output_shape = get_output_shape(attrs, input.shape);

  SimTaskBinding fwd_binding;
  fwd_binding.bind(INPUT, input);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind_arg(PROFILING, settings);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor = env.get_fwd_accessor(SPLIT_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(SPLIT_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<SPLIT_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_input_slot(INPUT);
  init.add_output_slot(OUTPUT);
  // TODO Note: split operator does not need SplitDeviceState, how to register
  // the init_task and how to implement the init_task like cast OP?
}

template <>
void register_task<SPLIT_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  register_task(SPLIT_FWD_TASK_ID, "Split Fwd", fwd, forward_task);
}

template <>
void register_task<SPLIT_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(get_op_signature(SPLIT_FWD_TASK_ID));

  register_task(SPLIT_BWD_TASK_ID, "Split Bwd", bwd, backward_task);
}

}; // namespace FlexFlow

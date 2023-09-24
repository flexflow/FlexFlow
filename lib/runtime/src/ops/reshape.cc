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

#include "reshape.h"
#include "kernels/reshape_kernels.h"
#include "legion/legion_utilities.h"
#include "utils/exception.h"
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

using namespace FlexFlow::Kernels::Reshape;

/* Params */
bool operator==(ReshapeParams const &lhs, ReshapeParams const &rhs) {
  return lhs.shape == rhs.shape;
}

bool ReshapeParams::is_valid(ParallelTensorShape const &input) const {
  return input.is_valid();
}

enum slots { INPUT, OUTPUT, ATTRS, PROFILING, PER_DEVICE_STATE };

OpTaskInvocation init(ReshapeAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(INPUT, input_parallel_tensor_shape(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {RESHAPE_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(ReshapeAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PER_DEVICE_STATE, per_device_op_state<ReshapePerDeviceState>());
  binding.bind_arg(PROFILING, profiling_settings());

  binding.bind(INPUT, input_parallel_tensor_shape(0));
  binding.bind(OUTPUT, output_tensor(0));
  return {RESHAPE_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(ReshapeAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {RESHAPE_BWD_TASK_ID, binding};
}

static DeviceSpecific<ReshapePerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {
  NOT_IMPLEMENTED();
}

static DeviceSpecific<ReshapePerDeviceState>
    init_task(Task const *task,
              std::vector<PhysicalRegion> const &regions,
              Context ctx,
              Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  return init_task_impl(acc);
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
         acc.get_argument<ReshapePerDeviceState>(PER_DEVICE_STATE);
  Profiling profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 "[Reshape] forward time = %.2lfms\n",
                 per_device_state,
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
  auto per_device_state =
      acc.get_argument<DeviceSpecific<ReshapePerDeviceState>>(PER_DEVICE_STATE);
  Profiling profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input_grad = acc.get_tensor<Permissions::RO>(INPUT);
  auto output_grad = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 "[Reshape] backward time = %.2lfms\n",
                 per_device_state,
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
                                  ReshapeAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view) {

  // reshape has no cost
  // Note(lamda):if reshape has cost, we can optimize this implementation

  float forward_time = 0.0;
  float backward_time = 0.0;
  float sync_time = 0.0;
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<RESHAPE_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_input_slot(INPUT);
  init.add_output_slot(OUTPUT);

  register_task(RESHAPE_INIT_TASK_ID, "Reshape Init", init, init_task);
}

template <>
void register_task<RESHAPE_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_unchecked_arg_slot<ReshapePerDeviceState>(PER_DEVICE_STATE);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  register_task(RESHAPE_FWD_TASK_ID, "Reshape Fwd", fwd, forward_task);
}

template <>
void register_task<RESHAPE_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_binding(get_op_signature(RESHAPE_FWD_TASK_ID));

  register_task(RESHAPE_BWD_TASK_ID, "Reshape Bwd", bwd, backward_task);
}

}; // namespace FlexFlow

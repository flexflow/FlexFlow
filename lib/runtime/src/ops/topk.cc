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

#include "topk.h"
#include "kernels/topk_kernels.h"
#include "op-attrs/get_output_shapes.h"
#include "utils/exception.decl.h"

namespace FlexFlow {
// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::InlineLauncher;
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
using PCG::Node;

using namespace FlexFlow::Kernels::TopK;

// For an input tensor, computes the top k entries in each row
// (resp. vector along the last dimension). Thus,
// values.shape = indices.shape = input.shape[:-1] + [k]

enum Slots { INPUT, OUTPUT, ATTRS, PROFILING, PER_DEVICE_STATE };

OpTaskInvocation init(TopKAttrs const &attrs) {
  OpTaskBinding binding;

  bind.bind_arg(ATTRS, attrs); // Note: we just bind SplitAttrs here, for init_task_impl

  return {TOPK_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(TopKAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PER_DEVICE_STATE, per_device_op_state<TopKPerDeviceState>());
  binding.bind_arg(PROFILING, profiling_settings());
  bind.bind_arg(ATTRS, attrs);

  binding.bind(INPUT, input_parallel_tensor_shape(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {TOPK_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(TopKAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {TOPK_BWD_TASK_ID, binding};
}

static DeviceSpecific<TopKPerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {

  auto attrs = acc.get_argument<TopKAttrs>(ATTRS);

  DeviceSpecific<TopKPerDeviceState> per_device_state =
      acc.create_device_specific<TopKPerDeviceState>(init_kernel(attrs.sorted));
  return per_device_state;
}

static DeviceSpecific<TopKPerDeviceState>
    init_task(Task const *task,
              std::vector<PhysicalRegion> const &regions,
              Context ctx,
              Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  return init_task_impl(acc);
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto attrs = acc.get_argument<TopKAttrs>(ATTRS);
  auto per_device_state =
      acc.get_device_specific<TopKPerDeviceState>(PER_DEVICE_STATE);
  auto profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  // Note: this may have problem, we can fix later, the below code is copy from
  // old code we use these elements to get length, batch_size, Question: hwo to
  // get index_ptr?
  Task *task = acc.task;
  Context ctx = acc.ctx;
  Runtime *runtime = acc.runtime;
  Domain in1_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain out1_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Domain out2_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());

  int in_cols = in1_domain.hi()[0] - in1_domain.lo()[0] + 1;
  int out1_cols = out1_domain.hi()[0] - out1_domain.lo()[0] + 1;
  int out2_cols = out2_domain.hi()[0] - out2_domain.lo()[0] + 1;

  assert(out1_domain == out2_domain);
  for (int i = 1; i < in1_domain.get_dim(); i++) {
    assert(in1_domain.lo()[i] == out1_domain.lo()[i]);
    assert(in1_domain.hi()[i] == out1_domain.hi()[i]);
  }

  int *index_ptr = helperGetTensorPointerWO<int>(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);

  int length = in1_domain.hi()[0] - in1_domain.lo()[0] + 1;
  size_t batch_size = in1_domain.get_volume() / length;

  return profiling(forward_kernel,
                   profiling,
                   "[TopK] forward_time = %.2lfms\n",
                   per_device_state,
                   input.get_float_ptr(),
                   output.get_float_ptr(),
                   index_ptr,
                   batch_size,
                   length,
                   attrs.k,
                   attrs.sorted);
}

static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  forward_task_impl(acc);
}

static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  auto attrs = acc.get_argument<TopKAttrs>(ATTRS);
  auto per_device_state =
      acc.get_device_specific<TopKPerDeviceState>(PER_DEVICE_STATE);
  auto profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input_grad = acc.get_tensor_grad<Permissions::RO>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::WO>(OUTPUT);

  // Note: this may have problem, we can fix later, the below code is copy from
  // old code
  Task *task = acc.task;
  Context ctx = acc.ctx;
  Runtime *runtime = acc.runtime;

  Domain out1_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain out2_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  assert(out1_domain == out2_domain);
  for (int i = 1; i < in_domain.get_dim(); i++) {
    assert(in_domain.lo()[i] == out1_domain.lo()[i]);
    assert(in_domain.hi()[i] == out1_domain.hi()[i]);
  }

  // Question: what's the indice_ptr, I think the value_grad_ptr is output_grad
  int const *indices_ptr = helperGetTensorPointerRO<int>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  int length = in_domain.hi()[0] - in_domain.lo()[0] + 1;
  size_t batch_size = in_domain.get_volume() / length;

  return profiling(backward_kernel,
                   profiling,
                   "[TopK] backward_time = %.2lfms\n",
                   per_device_state,
                   output_grad.get_float_ptr(),
                   indices_ptr,
                   input_grad.get_float_ptr(),
                   batch_size,
                   length,
                   attrs.k);
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  TopKAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view) {
  auto env = sim.new_environment();

  ParallelTensorShape output_shape = get_output_shapes(input, attrs);

  SimTaskBinding init_binding;
  init_binding.bind_arg(ATTRS, attrs);

  auto init_accessor = env.get_init_accessor(TOPK_INIT_TASK_ID, init_binding);
  DeviceSpecific<TopKPerDeviceState> per_device_state =
      init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;
  fwd_binding.bind_arg(PER_DEVICE_STATE,
                       per_device_op_state<TopKPerDeviceState>());
  fwd_binding.bind_arg(PROFILING, profiling_settings());
  fwd_binding.bind(INPUT, input_tensor(0));
  fwd_binding.bind(OUTPUT, output_tensor(0));
  fwd_binding.bind_arg(ATTRS, attrs);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor = env.get_fwd_accessor(TOPK_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(TOPK_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<TOPK_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_arg_slot<SplitAttrs>(ATTRS); // Note: this may have some question

  register_task(SPLIT_INIT_TASK_ID, "Split Init", init, init_task);
}

template <>
void register_task<TOPK_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  init.add_arg_slot<ProfilingSettings>(PROFILING);
  init.add_arg_slot<SplitAttrs>(ATTRS); // Note: this may have some question
  init.add_unchecked_arg_slot<SplitPerDeviceState>(PER_DEVICE_STATE);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  register_task(SPLIT_FWD_TASK_ID, "Split Forward", fwd, forward_task);
}

template <>
void register_task<TOPK_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(get_op_signature(SPLIT_FWD_TASK_ID));

  register_task(SPLIT_BWD_TASK_ID, "Split Backward", bwd, backward_task);
}

}; // namespace FlexFlow

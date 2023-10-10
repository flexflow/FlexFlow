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

#include "reverse.h"
#include "kernels/accessor.h"
#include "kernels/reverse_kernels.h"
#include "op-attrs/get_output_shapes.h"

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

using namespace FlexFlow::Kernels::Reverse;

enum Slots { INPUT, OUTPUT, ATTRS, PROFILING };

OpTaskInvocation forward(ReverseAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PROFILING, profiling_settings());
  bind.bind_arg(ATTRS, attrs);

  binding.bind(INPUT, input_parallel_tensor_shape(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {REVERSE_FWD_TASK_ID, binding};
}
OpTaskInvocation backward(ReverseAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {REVERSE_BWD_TASK_ID, binding};
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto attrs = acc.get_argument<ReverseAttrs>(ATTRS);

  int output_size = outtput.shape.get_volume();
  auto axis = attrs.axis;
  coord_t in_blk_size = 1, reverse_dim_size = 1, num_out_blks = 1;
  for (int i = 0; i < output.shape.get_dim(); i++) {
    if (i < axis) {
      in_blk_size *= output.shape[i];
    } else if (i == axis) {
      reverse_dim_size = output.shape[i];
    } else {
      num_out_blks *= output.shape[i];
    }
  }

  return profile(forward_kernel,
                 profiling,
                 "[reverse] forward_time = %.2lfms\n",
                 input.get_float_ptr(),
                 output.get_float_ptr(),
                 num_out_blks,
                 reverse_dim_size,
                 in_blk_size,
                 output_size);
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
  auto attrs = acc.get_argument<ReverseAttrs>(ATTRS);

  int axis = input.shape.get_dim() - attrs.axis - 1;
  coord_t in_blk_size = 1, reverse_dim_size = 1, num_out_blks = 1;
  for (int i = 0; i < input_grad.shape.get_dim(); i++) {
    if (i < axis) {
      in_blk_size *= input_grad.shape[i];
    } else if (i == axis) {
      reverse_dim_size = input_grad.shape[i];
    } else {
      num_out_blks *= input_grad.shape[i];
    }
  }

  return profile(backward_kernel,
                 profiling,
                 "[reverse] backward_time = %.2lfms\n",
                 output_grad.get_float_ptr(),
                 input_grad.get_float_ptr(),
                 num_out_blks,
                 reverse_dim_size,
                 in_blk_size,
                 input.shape.get_volume());
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ReverseAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view) {
  auto env = sim.new_environment();

  SimTaskBinding fwd_binding;

  ParallelTensorShape output_shape = get_output_shape(attrs, input.shape);

  fwd_binding.bind(INPUT, input.shape);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind_arg(ATTRS, attrs);

  auto fwd_accessor = env.get_fwd_accessor(REVERSE_FWD_TASK_ID, fwd_binding);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);
  // TODO Note: what should bwd_bindinig  bind?
  // according to aggregate.cc, bwd_binding bind the full_gate_gradients and
  // true_gate_assign
  auto bwd_accessor = env.get_bwd_accessor(REVERSE_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);

  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<REVERSE_FWD_TASK_ID>()) {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<bool>(PROFILING);
  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  register_task(REVERSE_FWD_TASK_ID, "Reverse forward", fwd, forward_task);
}

template <>
void register_task<REVERSE_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(get_op_signature(REVERSE_BWD_TASK_ID));
  register_task(REVERSE_BWD_TASK_ID, "Reverse backward", bwd, backward_task);
}

}; // namespace FlexFlow

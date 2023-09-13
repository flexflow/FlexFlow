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

#include "batch_matmul.h"
#include "kernels/batch_matmul_kernels.h"
#include "legion.h"
#include "op-attrs/get_output_shapes.h"
#include "op-attrs/ops/batch_matmul.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::BatchMatmul;

using Legion::Context;
using Legion::PhysicalRegion;
using Legion::Runtime;
using Legion::Task;

enum Slots {
  A_INPUT, // tensor
  B_INPUT, // tensor
  ATTRS,
  OUTPUT,  // tensor
  PROFILING,
  HANDLE,
  PER_DEVICE_STATE,
  ITERATION_CONFIG
};

OpTaskInvocation init(BatchMatmulAttrs const &attrs) {
  OpTaskBinding init;
  init.bind_arg(HANDLE, ff_handle());

  return {BATCHMATMUL_INIT_TASK_ID, init};
}

OpTaskInvocation forward(BatchMatmulAttrs const &attrs) {
  OpTaskBinding fwd;

  fwd.bind(A_INPUT, input_tensor(0));
  fwd.bind(B_INPUT, input_tensor(1));
  fwd.bind(OUTPUT, output_tensor(0));

  fwd.bind_arg(ATTRS, attrs);

  fwd.bind_arg(PROFILING, profiling_settings());
  fwd.bind_arg(PER_DEVICE_STATE, per_device_op_state<BMMPerDeviceState>());
  fwd.bind_arg(ITERATION_CONFIG, iteration_config());

  return {BATCHMATMUL_FWD_TASK_ID, fwd};
}

OpTaskInvocation backward(BatchMatmulAttrs const &attrs) {
  OpTaskBinding bwd = infer_bwd_binding(forward(attrs).binding);

  return {BATCHMATMUL_BWD_TASK_ID, bwd};
}

static DeviceSpecific<BMMPerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);
  Allocator allocator = acc.get_allocator();

  DeviceSpecific<BMMPerDeviceState> per_device_state =
      acc.create_device_specific<BMMPerDeviceState>(
          init_kernel(handle, allocator));

  return per_device_state;
}

static DeviceSpecific<BMMPerDeviceState>
    init_task(Task const *task,
              std::vector<PhysicalRegion> const &regions,
              Context ctx,
              Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  return init_task_impl(acc);
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto a_input = acc.get_tensor<Permissions::RO>(A_INPUT);
  auto b_input = acc.get_tensor<Permissions::RO>(B_INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto attrs = acc.get_argument<BatchMatmulAttrs>(ATTRS);

  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto per_device_state = acc.get_argument<BMMPerDeviceState>(PER_DEVICE_STATE);
  FFIterationConfig iter_config =
      acc.get_argument<FFIterationConfig>(ITERATION_CONFIG);

  int m = b_input.shape[legion_dim_t(0)];
  assert(m == output.shape[legion_dim_t(0)]);
  int n = a_input.shape[legion_dim_t(1)];
  assert(n == output.shape[legion_dim_t(1)]);
  int k = a_input.shape[legion_dim_t(0)];
  assert(k == b_input.shape[legion_dim_t(1)]);

  assert(a_input.shape.get_volume() == b_input.shape.get_volume());
  assert(a_input.shape.get_volume() == output.shape.get_volume());

  int batch = 1;
  for (int i = 2; i < a_input.shape.get_dim();
       i++) { // get_dim() or get_volume()?
    int dim_size = a_input.shape[legion_dim_t(i)];
    assert(dim_size == b_input.shape[legion_dim_t(i)]);
    assert(dim_size == output.shape[legion_dim_t(i)]);
    batch *= dim_size;
  }

  return profile(forward_kernel,
                 profiling,
                 "[BatchMatmul] forward_time = %.2lfms\n",
                 per_device_state,
                 output.get_float_ptr(),
                 a_input.get_float_ptr(),
                 b_input.get_float_ptr(),
                 m,
                 n,
                 k,
                 batch,
                 iter_config.seq_length,
                 attrs.a_seq_length_dim,
                 attrs.b_seq_length_dim);
}

static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  forward_task_impl(acc);
}

static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  // BatchMatmul* bmm = (BatchMatmul*) task->args;
  FFIterationConfig iter_config =
      acc.get_argument<FFIterationConfig>(ITERATION_CONFIG);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto per_device_state = acc.get_argument<BMMPerDeviceState>(PER_DEVICE_STATE);

  auto output = acc.get_tensor<Permissions::RO>(OUTPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RW>(OUTPUT);
  // is this equivalent to checking `Domain` equality?
  assert(output.shape == output_grad.shape);

  auto a_input = acc.get_tensor<Permissions::RO>(A_INPUT);
  auto a_input_grad = acc.get_tensor_grad<Permissions::RW>(A_INPUT);
  assert(a_input.shape == a_input_grad.shape);

  auto b_input = acc.get_tensor<Permissions::RO>(B_INPUT);
  auto b_input_grad = acc.get_tensor_grad<Permissions::RW>(B_INPUT);
  assert(b_input.shape == b_input_grad.shape);

  // check dins
  int m = b_input.shape[legion_dim_t(0)];
  assert(m == output.shape[legion_dim_t(0)]);
  int n = a_input.shape[legion_dim_t(1)];
  assert(n == output.shape[legion_dim_t(1)]);
  int k = a_input.shape[legion_dim_t(0)];
  assert(k == b_input.shape[legion_dim_t(1)]);
  assert(a_input.shape.get_volume() == b_input.shape.get_volume());
  assert(a_input.shape.get_volume() == output.shape.get_volume());
  int batch = 1;
  for (int i = 2; i < a_input.shape.dims.num_dims(); i++) {
    int dim_size = a_input.shape[legion_dim_t(i)];
    assert(dim_size == b_input.shape[legion_dim_t(i)]);
    assert(dim_size == output.shape[legion_dim_t(i)]);
    batch *= dim_size;
  }

  return profile(backward_kernel,
                 profiling,
                 "[BatchMatmul] backward_time = %.2lfms\n",
                 per_device_state,
                 output.get_float_ptr(),
                 output_grad.get_float_ptr(),
                 a_input.get_float_ptr(),
                 a_input_grad.get_float_ptr(),
                 b_input.get_float_ptr(),
                 b_input_grad.get_float_ptr(),
                 m,
                 n,
                 k,
                 batch);
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  BatchMatmulAttrs const &attrs,
                                  InputParallelTensorDesc const &a_input,
                                  InputParallelTensorDesc const &b_input,
                                  ProfilingSettings const &settings,
                                  MachineView const &pc) {
  auto env = sim.new_environment();

  ParallelTensorShape output_shape =
      get_output_shape(attrs, a_input.shape, b_input.shape);

  SimTaskBinding init_binding;
  init_binding.bind_arg(HANDLE, ff_handle());

  auto init_accessor =
      env.get_init_accessor(BATCHMATMUL_INIT_TASK_ID, init_binding);
  DeviceSpecific<BMMPerDeviceState> per_device_state =
      init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;
  fwd_binding.bind(A_INPUT, a_input);
  fwd_binding.bind(B_INPUT, b_input);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind_arg(ATTRS, attrs);
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind_arg(PER_DEVICE_STATE, per_device_state);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor =
      env.get_fwd_accessor(BATCHMATMUL_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor =
      env.get_bwd_accessor(BATCHMATMUL_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<BATCHMATMUL_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);
  init.add_return_value<BMMPerDeviceState>();

  register_task(BATCHMATMUL_INIT_TASK_ID, "BatchMatmul Init", init, init_task);
}

template <>
void register_task<BATCHMATMUL_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(A_INPUT);
  fwd.add_input_slot(B_INPUT);
  fwd.add_output_slot(OUTPUT);
  fwd.add_arg_slot<BatchMatmulAttrs>(ATTRS);
  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_unchecked_arg_slot<BMMPerDeviceState>(PER_DEVICE_STATE);

  register_task(BATCHMATMUL_FWD_TASK_ID, "BatchMatmul Fwd", fwd, forward_task);
}

template <>
void register_task<BATCHMATMUL_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(get_op_signature(ATTENTION_FWD_TASK_ID));

  register_task(BATCHMATMUL_BWD_TASK_ID, "BatchMatmul Bwd", bwd, backward_task);
}

}; // namespace FlexFlow

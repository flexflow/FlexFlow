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

#include "attention.h"
#include "kernels/attention_kernels.h"
#include "legion.h"
#include "op-attrs/ops/attention.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::MultiHeadAttention;

using Legion::Context;
using Legion::PhysicalRegion;
using Legion::Runtime;
using Legion::Task;

enum Slots {
  QUERY_PARALLEL_TENSOR_SHAPE,
  KEY_PARALLEL_TENSOR_SHAPE,
  VALUE_PARALLEL_TENSOR_SHAPE,
  QPROJSIZE,
  KPROJSIZE,
  VPROJSIZE,
  OPROJSIZE,
  ATTRS,
  PROFILING,
  QUERY,
  KEY,
  VALUE,
  WEIGHTS,
  OUTPUT,
  HANDLE,
  PER_DEVICE_STATE
};

OpTaskInvocation init(MultiHeadAttentionAttrs const &attrs) {
  OpTaskBinding b;

  b.bind_arg(HANDLE, ff_handle());
  b.bind_arg(ATTRS, attrs);

  b.bind_arg(QUERY_PARALLEL_TENSOR_SHAPE, input_parallel_tensor_shape(0));
  b.bind_arg(KEY_PARALLEL_TENSOR_SHAPE, input_parallel_tensor_shape(1));
  b.bind_arg(VALUE_PARALLEL_TENSOR_SHAPE, input_parallel_tensor_shape(2));

  b.bind_arg(QPROJSIZE, get_qProjSize(attrs));
  b.bind_arg(KPROJSIZE, get_kProjSize(attrs));
  b.bind_arg(VPROJSIZE, get_vProjSize(attrs));
  b.bind_arg(OPROJSIZE, get_oProjSize(attrs));

  return {ATTENTION_INIT_TASK_ID, b};
}

OpTaskInvocation forward(MultiHeadAttentionAttrs const &attrs) {
  OpTaskBinding b;

  b.bind(QUERY, input_tensor(0));
  b.bind(KEY, input_tensor(1));
  b.bind(VALUE, input_tensor(2));
  b.bind(WEIGHTS, weight_tensor(0));
  b.bind(OUTPUT, output_tensor(0));

  b.bind_arg(PROFILING, profiling_settings());
  b.bind_arg(PER_DEVICE_STATE, per_device_op_state<MHAPerDeviceState>());

  return {ATTENTION_FWD_TASK_ID, b};
}

OpTaskInvocation backward(MultiHeadAttentionAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {ATTENTION_BWD_TASK_ID, b};
}

static DeviceSpecific<MHAPerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto const &attrs = acc.get_argument<MultiHeadAttentionAttrs>(ATTRS);
  Allocator allocator = acc.get_allocator();
  int qProjSize = acc.get_argument<int>(QPROJSIZE);
  int kProjSize = acc.get_argument<int>(KPROJSIZE);
  int vProjSize = acc.get_argument<int>(VPROJSIZE);
  int oProjSize = acc.get_argument<int>(OPROJSIZE);
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);
  ParallelTensorShape query_parallel_tensor_shape =
      acc.get_argument<ParallelTensorShape>(QUERY_PARALLEL_TENSOR_SHAPE);
  ParallelTensorShape key_parallel_tensor_shape =
      acc.get_argument<ParallelTensorShape>(KEY_PARALLEL_TENSOR_SHAPE);
  ParallelTensorShape value_parallel_tensor_shape =
      acc.get_argument<ParallelTensorShape>(VALUE_PARALLEL_TENSOR_SHAPE);

  MultiHeadAttentionInputs<ParallelTensorShape> inputs =
      MultiHeadAttentionInputs<ParallelTensorShape>(
          query_parallel_tensor_shape,
          key_parallel_tensor_shape,
          value_parallel_tensor_shape);

  ParallelTensorShape output_parallel_tensor_shape =
      get_output_shape(attrs, inputs);
  ParallelTensorShape weight_parallel_tensor_shape =
      get_weights_shape(attrs, inputs);

  int kvSeqLength = get_kvSeqLength(inputs);
  int qSize = get_qSize(inputs);
  int kSize = get_kSize(inputs);
  int vSize = get_vSize(inputs);

  int qoSeqLength = get_piece_shape(query_parallel_tensor_shape)[ff_dim_t(1)];
  int num_samples = get_piece_shape(query_parallel_tensor_shape)[ff_dim_t(2)];
  int num_heads = get_piece_shape(weight_parallel_tensor_shape)[ff_dim_t(1)];

  assert(qoSeqLength == query.shape[legion_dim_t(1)]);
  assert(qSize == query.shape[legion_dim_t(0)]);
  assert(num_samples == key.shape[legion_dim_t(2)]);
  assert(kvSeqLength == key.shape[legion_dim_t(1)]);
  assert(kSize == key.shape[legion_dim_t(0)]);
  assert(num_samples == value.shape[legion_dim_t(2)]);
  assert(kvSeqLength == value.shape[legion_dim_t(1)]);
  assert(vSize == value.shape[legion_dim_t(0)]);
  assert(num_samples == output.shape[legion_dim_t(2)]);
  assert(qoSeqLength == output.shape[legion_dim_t(1)]);
  assert(oProjSize == output.shape[legion_dim_t(0)]);

  DeviceSpecific<MHAPerDeviceState> per_device_state =
      acc.create_device_specific<MHAPerDeviceState>(
          init_kernel(handle,
                      allocator,
                      num_samples,
                      num_heads,
                      qSize,
                      kSize,
                      vSize,
                      qProjSize,
                      kProjSize,
                      vProjSize,
                      oProjSize,
                      qoSeqLength,
                      kvSeqLength,
                      attrs.add_bias_kv));

  assert(weight.shape.get_volume() * sizeof(float) ==
         acc.unwrap(per_device_state)->weightSize);
  return per_device_state;
}

static DeviceSpecific<MHAPerDeviceState>
    init_task(Task const *task,
              std::vector<PhysicalRegion> const &regions,
              Context ctx,
              Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  return init_task_impl(acc);
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto query = acc.get_tensor<Permissions::RO>(QUERY);
  auto key = acc.get_tensor<Permissions::RO>(KEY);
  auto value = acc.get_tensor<Permissions::RO>(VALUE);
  auto weight = acc.get_tensor<Permissions::RO>(WEIGHTS);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto per_device_state = acc.get_argument<MHAPerDeviceState>(PER_DEVICE_STATE);

  return profile(forward_kernel,
                 profiling,
                 "[MultiHeadAttention] forward_time = %.2lfms\n",
                 per_device_state,
                 query.get_float_ptr(),
                 key.get_float_ptr(),
                 value.get_float_ptr(),
                 weight.get_float_ptr(),
                 output.get_float_ptr());
}

static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  forward_task_impl(acc);
}

static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  auto query = acc.get_tensor<Permissions::RO>(QUERY);
  auto key = acc.get_tensor<Permissions::RO>(KEY);
  auto value = acc.get_tensor<Permissions::RO>(VALUE);
  auto weight = acc.get_tensor<Permissions::RO>(WEIGHTS);

  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);
  auto weight_grad = acc.get_tensor_grad<Permissions::RW>(WEIGHTS);
  auto query_grad = acc.get_tensor_grad<Permissions::RW>(QUERY);
  auto key_grad = acc.get_tensor_grad<Permissions::RW>(KEY);
  auto value_grad = acc.get_tensor_grad<Permissions::RW>(VALUE);

  auto per_device_state = acc.get_argument<MHAPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  float *key_grad_ptr =
      (key_grad == query_grad) ? nullptr : key_grad.get_float_ptr();
  float *value_grad_ptr = (value_grad == query_grad || value_grad == key_grad)
                              ? nullptr
                              : value_grad.get_float_ptr();

  assert(value_grad.shape == value.shape);
  assert(key_grad.shape == key.shape);

  assert(query_grad.shape == query.shape);
  assert(weight_grad.shape.get_volume() == weight.shape.get_volume());

  return profile(backward_kernel,
                 profiling,
                 "[MultiHeadAttention] backward_time = %.2lfms\n",
                 per_device_state,
                 query.get_float_ptr(),
                 query_grad.get_float_ptr(),
                 key.get_float_ptr(),
                 key_grad_ptr,
                 value.get_float_ptr(),
                 value_grad_ptr,
                 weight.get_float_ptr(),
                 weight_grad.get_float_ptr(),
                 output_grad.get_float_ptr());
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  MultiHeadAttentionAttrs const &attrs,
                                  InputParallelTensorDesc const &query_shape,
                                  InputParallelTensorDesc const &key_shape,
                                  InputParallelTensorDesc const &value_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv) {
  auto env = sim.new_environment();

  MultiHeadAttentionInputs<ParallelTensorShape> inputs =
      MultiHeadAttentionInputs<ParallelTensorShape>(
          query_shape.shape, key_shape.shape, value_shape.shape);
  ParallelTensorShape output_shape = get_output_shape(attrs, inputs);
  ParallelTensorShape weight_shape = get_weights_shape(attrs, inputs);

  SimTaskBinding init_binding;
  init_binding.bind_arg(HANDLE, ff_handle());
  init_binding.bind_arg(ATTRS, attrs);
  init_binding.bind_arg(QUERY_PARALLEL_TENSOR_SHAPE,
                        input_parallel_tensor_shape(0));
  init_binding.bind_arg(KEY_PARALLEL_TENSOR_SHAPE,
                        input_parallel_tensor_shape(1));
  init_binding.bind_arg(VALUE_PARALLEL_TENSOR_SHAPE,
                        input_parallel_tensor_shape(2));
  init_binding.bind_arg(QPROJSIZE, get_qProjSize(attrs));
  init_binding.bind_arg(KPROJSIZE, get_kProjSize(attrs));
  init_binding.bind_arg(VPROJSIZE, get_vProjSize(attrs));
  init_binding.bind_arg(OPROJSIZE, get_oProjSize(attrs));

  auto init_accessor =
      env.get_init_accessor(ATTENTION_INIT_TASK_ID, init_binding);
  DeviceSpecific<MHAPerDeviceState> per_device_state =
      init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;
  fwd_binding.bind(QUERY, query_shape);
  fwd_binding.bind(KEY, key_shape);
  fwd_binding.bind(VALUE, value_shape);
  fwd_binding.bind(WEIGHTS, weight_shape);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind_arg(PER_DEVICE_STATE, per_device_state);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor = env.get_fwd_accessor(ATTENTION_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(ATTENTION_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<ATTENTION_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);
  init.add_arg_slot<ParallelTensorShape>(QUERY_PARALLEL_TENSOR_SHAPE);
  init.add_arg_slot<ParallelTensorShape>(KEY_PARALLEL_TENSOR_SHAPE);
  init.add_arg_slot<ParallelTensorShape>(VALUE_PARALLEL_TENSOR_SHAPE);
  init.add_arg_slot<int>(QPROJSIZE);
  init.add_arg_slot<int>(KPROJSIZE);
  init.add_arg_slot<int>(VPROJSIZE);
  init.add_arg_slot<int>(OPROJSIZE);
  init.add_arg_slot<MultiHeadAttentionAttrs>(ATTRS);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  init.add_return_value<MHAPerDeviceState>();

  register_task(
      ATTENTION_INIT_TASK_ID, "MultiHeadAttention Init", init, init_task);
}

template <>
void register_task<ATTENTION_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(QUERY);
  fwd.add_input_slot(KEY);
  fwd.add_input_slot(VALUE);
  fwd.add_weight_slot(WEIGHTS);
  fwd.add_output_slot(OUTPUT);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_unchecked_arg_slot<MHAPerDeviceState>(PER_DEVICE_STATE);

  register_task(
      ATTENTION_FWD_TASK_ID, "MultiHeadAttention Fwd", fwd, forward_task);
}

template <>
void register_task<ATTENTION_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(get_op_signature(ATTENTION_FWD_TASK_ID));

  register_task(
      ATTENTION_BWD_TASK_ID, "MultiHeadAttention Bwd", bwd, backward_task);
}

} // namespace FlexFlow

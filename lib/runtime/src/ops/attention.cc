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

namespace FlexFlow {

using namespace FlexFlow::Kernels::MultiHeadAttention;

using Legion::Task;
using Legion::Context;
using Legion::PhysicalRegion;
using Legion::Runtime;

enum Slots {
  ATTRS,
  PROFILING,
  KVSEQLENGTH,
  QSIZE,
  KSIZE,
  VSIZE,
  QPROJSIZE,
  KPROJSIZE,
  VPROJSIZE,
  OPROJSIZE,
  QUERY,
  KEY,
  VALUE,
  WEIGHTS,
  OUTPUT,
  PER_DEVICE_STATE
};

OpTaskInvocation init(MultiHeadAttentionAttrs const &attrs) {
  OpTaskBinding b;

  b.bind(QUERY, input_tensor(0));
  b.bind(KEY, input_tensor(1));
  b.bind(VALUE, input_tensor(2));
  b.bind(WEIGHTS, weight_tensor(0));
  b.bind(OUTPUT, output_tensor(0));

  b.bind_arg(ATTRS, attrs);
  b.bind_arg(PROFILING, profiling_settings());

  return {ATTENTION_INIT_TASK_ID, b};
}

OpTaskInvocation forward(MultiHeadAttentionAttrs const &attrs) {
  OpTaskBinding b;

  b.bind(QUERY, input_tensor(0));
  b.bind(KEY, input_tensor(1));
  b.bind(VALUE, input_tensor(2));
  b.bind(WEIGHTS, weight_tensor(0));
  b.bind(OUTPUT, output_tensor(0));
  b.bind_arg(PER_DEVICE_STATE, per_device_op_state<MHAPerDeviceState>());

  return {ATTENTION_FWD_TASK_ID, b};
}

OpTaskInvocation backward(MultiHeadAttentionAttrs const &attrs) {
  OpTaskBinding b;

  b.bind(QUERY, input_tensor(0));
  b.bind(KEY, input_tensor(1));
  b.bind(VALUE, input_tensor(2));
  b.bind_grad(QUERY, input_tensor(0));
  b.bind_grad(KEY, input_tensor(1));
  b.bind_grad(VALUE, input_tensor(2));
  b.bind(WEIGHTS, weight_tensor(0));
  b.bind_grad(WEIGHTS, weight_tensor(0));
  b.bind_grad(OUTPUT, output_tensor(0));

  return {ATTENTION_BWD_TASK_ID, b};
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  MultiHeadAttentionAttrs const &attrs,
                                  ParallelTensorShape const &query_shape,
                                  ParallelTensorShape const &key_shape,
                                  ParallelTensorShape const &value_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv) {
  auto env = sim.new_environment();

  auto query = allocate_input(env, query_shape);
  auto key = allocate_input(env, key_shape);
  auto value = allocate_input(env, value_shape);

  auto query_grad = allocate_input_grad(env, query_shape);
  auto key_grad = allocate_input_grad(env, key_shape);
  auto value_grad = allocate_input_grad(env, value_shape);

  MultiHeadAttentionInputs<ParallelTensorShape> input_shapes = {
      query_shape, key_shape, value_shape};

  ParallelTensorShape weight_shape = get_weights_shape(attrs, input_shapes);
  auto weights = allocate_weight(env, weight_shape);
  auto weights_grad = allocate_weight_grad(env, weight_shape);
  ParallelTensorShape output_shape = get_output_shape(attrs, input_shapes);
  auto output = allocate_output(env, output_shape);
  auto output_grad = allocate_output_grad(env, output_shape);

  MHAPerDeviceState per_device_state =
      init_kernel(get_ff_handle(env),
                  create_allocator(env),
                  get_num_samples(input_shapes),
                  attrs.num_heads,
                  get_qSize(input_shapes),
                  get_kSize(input_shapes),
                  get_vSize(input_shapes),
                  get_qProjSize(attrs),
                  get_kProjSize(attrs),
                  get_vProjSize(attrs),
                  get_oProjSize(attrs),
                  get_qoSeqLength(input_shapes),
                  get_kvSeqLength(input_shapes),
                  attrs.add_bias_kv);

  float forward_time = profiling_wrapper(forward_kernel,
                                         settings,
                                         per_device_state,
                                         get_float_ptr(query),
                                         get_float_ptr(key),
                                         get_float_ptr(value),
                                         get_float_ptr(weights),
                                         get_float_ptr(output))
                           .value();

  float backward_time = profiling_wrapper(backward_kernel,
                                          settings,
                                          per_device_state,
                                          get_float_ptr(query),
                                          get_float_ptr(query_grad),
                                          get_float_ptr(key),
                                          get_float_ptr(key_grad),
                                          get_float_ptr(value),
                                          get_float_ptr(value_grad),
                                          get_float_ptr(weights),
                                          get_float_ptr(weights_grad),
                                          get_float_ptr(output_grad))
                            .value();

  float sync_time = default_estimate_sync_time(env);

  return make_metrics(forward_time, backward_time, sync_time, env);
}

static DeviceSpecificArg<MHAPerDeviceState> *init_task(Task const *task,
                                    std::vector<PhysicalRegion> const &regions,
                                    Context ctx,
                                    Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  return init_task_impl(acc);
}

static DeviceSpecificArg<MHAPerDeviceState> *init_task_impl(TaskArgumentAccessor const &acc) {
  auto const &attrs = acc.get_argument<MultiHeadAttentionAttrs>(ATTRS);
  bool profiling = acc.get_argument<bool>(PROFILING);
  int kvSeqLength = acc.get_argument<int>(KVSEQLENGTH);
  int qSize = acc.get_argument<int>(QSIZE);
  int kSize = acc.get_argument<int>(KSIZE);
  int vSize = acc.get_argument<int>(VSIZE);
  int qProjSize = acc.get_argument<int>(QPROJSIZE);
  int kProjSize = acc.get_argument<int>(KPROJSIZE);
  int vProjSize = acc.get_argument<int>(VPROJSIZE);
  int oProjSize = acc.get_argument<int>(OPROJSIZE);
  Allocator allocator = acc.get_allocator();

  PerDeviceFFHandle handle = acc.get_per_device_ffhandle();

  auto query = acc.get_tensor<Permissions::RO>(QUERY);
  auto key = acc.get_tensor<Permissions::RO>(KEY);
  auto value = acc.get_tensor<Permissions::RO>(VALUE);
  auto weight = acc.get_tensor<Permissions::RO>(WEIGHTS);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  int qoSeqLength = query.shape[legion_dim_t(1)];

  int num_samples = query.shape[legion_dim_t(2)];
  assert(qoSeqLength == query.shape[legion_dim_t(1)]);
  assert(qSize == query.shape[legion_dim_t(0)]);
  assert(num_samples == key.shape[legion_dim_t(2)]);
  assert(kvSeqLength == key.shape[legion_dim_t(1)]);
  assert(kSize == key.shape[legion_dim_t(0)]);
  assert(num_samples == value.shape[legion_dim_t(2)]);
  assert(kvSeqLength == value.shape[legion_dim_t(1)]);
  assert(vSize == value.shape[legion_dim_t(0)]);
  int num_heads = weight.shape[legion_dim_t(1)];
  assert(num_samples == output.shape[legion_dim_t(2)]);
  assert(qoSeqLength == output.shape[legion_dim_t(1)]);
  assert(oProjSize(attrs) == output.shape[legion_dim_t(0)]);

  MHAPerDeviceState *m = new MHAPerDeviceState(init_kernel(handle,
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

  assert(weight.shape.get_volume() * sizeof(float) == m->weightSize);

  DeviceSpecificArg<MHAPerDeviceState> *n = new DeviceSpecificArg<MHAPerDeviceState>(m);
  return n;
}

static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);

  auto query = acc.get_tensor<FlexFlow::Permissions::RO>(QUERY);
  auto key = acc.get_tensor<FlexFlow::Permissions::RO>(KEY);
  auto value = acc.get_tensor<FlexFlow::Permissions::RO>(VALUE);
  auto weight = acc.get_tensor<FlexFlow::Permissions::RO>(WEIGHTS);
  auto output = acc.get_tensor<FlexFlow::Permissions::WO>(OUTPUT);
  auto per_device_state = acc.get_argument<MHAPerDeviceState>(PER_DEVICE_STATE);
  auto profiling_settings = acc.get_argument<ProfilingSettings>(PROFILING);

  profile(forward_kernel,
          profiling_settings,
          "[MultiHeadAttention] forward_time = %.2lfms\n",
          per_device_state,
          query.get_float_ptr(),
          key.get_float_ptr(),
          value.get_float_ptr(),
          weight.get_float_ptr(),
          output.get_float_ptr());
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);

  auto query = acc.get_tensor<FlexFlow::Permissions::RO>(QUERY);
  auto key = acc.get_tensor<FlexFlow::Permissions::RO>(KEY);
  auto value = acc.get_tensor<FlexFlow::Permissions::RO>(VALUE);
  auto weight = acc.get_tensor<FlexFlow::Permissions::RO>(WEIGHTS);
  auto per_device_state = acc.get_argument<MHAPerDeviceState>(PER_DEVICE_STATE);

  auto output_grad = acc.get_tensor_grad<FlexFlow::Permissions::RO>(OUTPUT);
  auto weight_grad = acc.get_tensor_grad<FlexFlow::Permissions::RW>(WEIGHTS);
  auto query_grad = acc.get_tensor_grad<FlexFlow::Permissions::RW>(QUERY);
  auto key_grad = acc.get_tensor_grad<FlexFlow::Permissions::RW>(KEY);
  auto value_grad = acc.get_tensor_grad<FlexFlow::Permissions::RW>(VALUE);
  auto profiling_settings = acc.get_argument<ProfilingSettings>(PROFILING);

  float *key_grad_ptr =
      (key_grad == query_grad) ? nullptr : key_grad.get_float_ptr();
  float *value_grad_ptr = (value_grad == query_grad || value_grad == key_grad)
                              ? nullptr
                              : value_grad.get_float_ptr();

  assert(value_grad.shape == value.shape);
  assert(key_grad.shape == key.shape);

  assert(query_grad.shape == query.shape);
  assert(weight_grad.shape.get_volume() == weight.shape.get_volume());

  profile(backward_kernel,
          profiling_settings,
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

template <>
void register_task<ATTENTION_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_arg_slot<MultiHeadAttentionAttrs>(ATTRS);
  init.add_arg_slot<bool>(PROFILING);
  init.add_arg_slot<int>(KVSEQLENGTH);
  init.add_arg_slot<int>(KSIZE);
  init.add_arg_slot<int>(VSIZE);
  init.add_arg_slot<int>(QPROJSIZE);

  init.add_input_slot(QUERY);
  init.add_input_slot(KEY);
  init.add_input_slot(VALUE);
  init.add_weight_slot(WEIGHTS);
  init.add_output_slot(OUTPUT);

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

  register_task(
      ATTENTION_FWD_TASK_ID, "MultiHeadAttention Fwd", fwd, forward_task);
}

template <>
void register_task<ATTENTION_BWD_TASK_ID>() {
  OpTaskSignature bwd(OpTaskType::BWD);

  bwd.add_input_slot(KEY);
  bwd.add_input_slot(QUERY);
  bwd.add_input_slot(VALUE);
  bwd.add_input_slot(VALUE);
  bwd.add_weight_slot(WEIGHTS);

  OpTensorSlotSpec key_gradient = OpTensorSlotSpec(
      KEY, FlexFlow::SlotType::TENSOR, FlexFlow::TensorRole::INPUT);
  key_gradient.is_grad = FlexFlow::IsGrad::YES;
  bwd.add_from_slot_spec(key_gradient);

  OpTensorSlotSpec query_gradient = OpTensorSlotSpec(
      QUERY, FlexFlow::SlotType::TENSOR, FlexFlow::TensorRole::INPUT);
  query_gradient.is_grad = FlexFlow::IsGrad::YES;
  bwd.add_from_slot_spec(query_gradient);

  OpTensorSlotSpec weights_gradient = OpTensorSlotSpec(
      WEIGHTS, FlexFlow::SlotType::TENSOR, FlexFlow::TensorRole::WEIGHT);
  weights_gradient.is_grad = FlexFlow::IsGrad::YES;
  bwd.add_from_slot_spec(weights_gradient);

  OpTensorSlotSpec output_gradient = OpTensorSlotSpec(
      OUTPUT, FlexFlow::SlotType::TENSOR, FlexFlow::TensorRole::OUTPUT);
  output_gradient.is_grad = FlexFlow::IsGrad::YES;
  bwd.add_from_slot_spec(output_gradient);

  register_task(
      ATTENTION_BWD_TASK_ID, "MultiHeadAttention Bwd", bwd, backward_task);
}

} // namespace FlexFlow

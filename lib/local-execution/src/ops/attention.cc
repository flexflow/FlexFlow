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
#include "local-execution/op_task_signature.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::MultiHeadAttention;

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

static DeviceSpecific<DeviceStates>
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

  MultiHeadAttentionInputs inputs = {
      shard_dim_at_idx(query_parallel_tensor_shape, ff_dim_t{0}).size,
      shard_dim_at_idx(query_parallel_tensor_shape, ff_dim_t{1}).size,
      qProjSize,
      kProjSize,
      vProjSize,
      query_parallel_tensor_shape.data_type};
  ;
  ParallelTensorShape weight_parallel_tensor_shape =
      throw_if_unexpected(get_weights_shape(attrs,
                                            query_parallel_tensor_shape,
                                            key_parallel_tensor_shape,
                                            value_parallel_tensor_shape));

  int kvSeqLength = get_kvSeqLength(inputs);
  int qSize = get_qSize(inputs);
  int kSize = get_kSize(inputs);
  int vSize = get_vSize(inputs);

  int qoSeqLength =
      dim_at_idx(get_piece_shape(query_parallel_tensor_shape), ff_dim_t(1));
  int num_samples =
      dim_at_idx(get_piece_shape(query_parallel_tensor_shape), ff_dim_t(2));
  int num_heads =
      dim_at_idx(get_piece_shape(weight_parallel_tensor_shape), ff_dim_t(1));

  MHAPerDeviceState per_device_state = init_kernel(handle,
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
                                                   attrs.add_bias_kv);
  return DeviceSpecific<DeviceStates>::create(per_device_state);
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto query = acc.get_tensor<Permissions::RO>(QUERY);
  auto key = acc.get_tensor<Permissions::RO>(KEY);
  auto value = acc.get_tensor<Permissions::RO>(VALUE);
  auto weight = acc.get_tensor<Permissions::RO>(WEIGHTS);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto per_device_state = acc.get_argument<MHAPerDeviceState>(PER_DEVICE_STATE);

  return profile(forward_kernel,
                 profiling,
                 "[MultiHeadAttention] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 query.get_float_ptr(),
                 key.get_float_ptr(),
                 value.get_float_ptr(),
                 weight.get_float_ptr(),
                 output.get_float_ptr());
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
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
                 "[MultiHeadAttention] backward_time = {:.2lf}ms\n",
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

TaskImplFunction get_attention_init_task_impl() {
  return init_task_impl;
}
TaskImplFunction get_attention_fwd_task_impl() {
  return forward_task_impl;
}
TaskImplFunction get_attention_bwd_task_impl() {
  return backward_task_impl;
}

OpTaskSignature get_attention_init_signature() {
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

  return init;
}

OpTaskSignature get_attention_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(QUERY);
  fwd.add_input_slot(KEY);
  fwd.add_input_slot(VALUE);
  fwd.add_weight_slot(WEIGHTS);
  fwd.add_output_slot(OUTPUT);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_unchecked_arg_slot<MHAPerDeviceState>(PER_DEVICE_STATE);

  return fwd;
}

OpTaskSignature get_attention_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_attention_fwd_signature());

  return bwd;
}

std::vector<task_id_t> get_task_ids(MultiHeadAttentionAttrs const &) {
  return {ATTENTION_INIT_TASK_ID, ATTENTION_FWD_TASK_ID, ATTENTION_BWD_TASK_ID};
}

} // namespace FlexFlow

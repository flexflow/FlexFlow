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

#include "task-spec/ops/impl/attention.h"
#include "kernels/attention_kernels.h"
#include "kernels/device_handle_t.dtg.h"
#include "op-attrs/ops/attention.h"
#include "op-attrs/ops/attention/multihead_attention_inputs.h"
#include "op-attrs/ops/attention/multihead_attention_parallel_inputs.h"
#include "task-spec/profiling.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::MultiHeadAttention;

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  MultiHeadAttentionAttrs attrs =
      acc.get_op_attrs().require_multi_head_attention();
  Allocator allocator = acc.get_allocator();

  DeviceType kernel_device_type = acc.get_kernel_device_type();

  positive_int qProjSize = get_qProjSize(attrs);
  positive_int kProjSize = get_kProjSize(attrs);
  positive_int vProjSize = get_vProjSize(attrs);
  positive_int oProjSize = get_oProjSize(attrs);

  device_handle_t handle = acc.get_ff_handle();

  TensorShape query_tensor_shape = acc.get_tensor_shape(TensorSlotName::QUERY);
  TensorShape key_tensor_shape = acc.get_tensor_shape(TensorSlotName::KEY);
  TensorShape value_tensor_shape = acc.get_tensor_shape(TensorSlotName::VALUE);

  MultiHeadAttentionInputs parsed =
      throw_if_unexpected(parse_attention_input_shape(
          query_tensor_shape, key_tensor_shape, value_tensor_shape));
  TensorShape weight_tensor_shape = throw_if_unexpected(get_weights_shape(
      attrs, query_tensor_shape, key_tensor_shape, value_tensor_shape));

  positive_int kvSeqLength = get_kvSeqLength(parsed);
  positive_int qSize = get_qSize(parsed);
  positive_int kSize = get_kSize(parsed);
  positive_int vSize = get_vSize(parsed);

  positive_int qoSeqLength = get_qoSeqLength(parsed);
  positive_int num_samples = get_num_samples(parsed);
  positive_int num_heads = attrs.num_heads;

  std::optional<MHAPerDeviceState> per_device_state = init_kernel(
      /*device_type=*/kernel_device_type,
      /*per_device_ff_handle=*/handle,
      /*allocator=*/allocator,
      /*num_samples=*/num_samples.int_from_positive_int(),
      /*num_heads=*/num_heads.int_from_positive_int(),
      /*qSize=*/qSize.int_from_positive_int(),
      /*kSize=*/kSize.int_from_positive_int(),
      /*vSize=*/vSize.int_from_positive_int(),
      /*qProjSize=*/qProjSize.int_from_positive_int(),
      /*kProjSize=*/kProjSize.int_from_positive_int(),
      /*vProjSize=*/vProjSize.int_from_positive_int(),
      /*oProjSize=*/oProjSize.int_from_positive_int(),
      /*qoSeqLength=*/qoSeqLength.int_from_positive_int(),
      /*kvSeqLength=*/kvSeqLength.int_from_positive_int(),
      /*add_bias_kv=*/attrs.add_bias_kv);

  return DeviceSpecificPerDeviceOpState{
      acc.make_device_specific(per_device_state),
  };
}

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  auto query = acc.get_tensor<Permissions::RO>(TensorSlotName::QUERY);
  auto key = acc.get_tensor<Permissions::RO>(TensorSlotName::KEY);
  auto value = acc.get_tensor<Permissions::RO>(TensorSlotName::VALUE);
  auto weight = acc.get_tensor<Permissions::RO>(TensorSlotName::WEIGHT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  std::optional<MHAPerDeviceState> per_device_state =
      acc.get_per_device_op_state().require_mha();

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[MultiHeadAttention] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 query.get_float_ptr(),
                 key.get_float_ptr(),
                 value.get_float_ptr(),
                 weight.get_float_ptr(),
                 output.get_float_ptr());
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto query = acc.get_tensor<Permissions::RO>(TensorSlotName::QUERY);
  auto key = acc.get_tensor<Permissions::RO>(TensorSlotName::KEY);
  auto value = acc.get_tensor<Permissions::RO>(TensorSlotName::VALUE);
  auto weight = acc.get_tensor<Permissions::RO>(TensorSlotName::WEIGHT);

  auto output_grad =
      acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);
  auto weight_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::WEIGHT);
  auto query_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::QUERY);
  auto key_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::KEY);
  auto value_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::VALUE);

  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  std::optional<MHAPerDeviceState> per_device_state =
      acc.get_per_device_op_state().require_mha();

  float *key_grad_ptr =
      (key_grad == query_grad) ? nullptr : key_grad.get_float_ptr();
  float *value_grad_ptr = (value_grad == query_grad || value_grad == key_grad)
                              ? nullptr
                              : value_grad.get_float_ptr();

  ASSERT(value_grad.shape == value.shape);
  ASSERT(key_grad.shape == key.shape);

  ASSERT(query_grad.shape == query.shape);
  ASSERT(weight_grad.shape == weight.shape);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
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
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}

TaskImplFunction get_attention_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_attention_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

} // namespace FlexFlow

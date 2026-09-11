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

#include "task-spec/ops/impl/concat.h"
#include "kernels/concat_kernels.h"
#include "op-attrs/tensor_slot_name.h"
#include "task-spec/profiling.h"
#include "task-spec/variadic_tensor_ref.h"
#include "utils/containers/slice.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Concat;

static std::vector<TensorSlotName> get_input_slots(ConcatAttrs const &attrs) {
  return slice(get_variadic_inputs_slot_name_sequence(),
               0,
               attrs.num_inputs.int_from_int_ge_two());
}

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  ConcatAttrs attrs = acc.get_op_attrs().require_concat();

  std::vector<TensorSlotName> input_slots = get_input_slots(attrs);

  std::vector<GenericTensorAccessorR> inputs =
      transform(input_slots,
                [&](TensorSlotName input_slot_name) -> GenericTensorAccessorR {
                  return acc.get_tensor<Permissions::RO>(input_slot_name);
                });

  ASSERT(inputs.size() <= MAX_NUM_INPUTS);

  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Concat] forward_time = {:.2lf}ms\n",
                 output,
                 inputs,
                 attrs.axis);
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  ConcatAttrs attrs = acc.get_op_attrs().require_concat();

  std::vector<TensorSlotName> input_slots = get_input_slots(attrs);

  std::vector<GenericTensorAccessorW> input_grads =
      transform(input_slots,
                [&](TensorSlotName input_slot_name) -> GenericTensorAccessorW {
                  return acc.get_tensor_grad<Permissions::RW>(input_slot_name);
                });

  ASSERT(input_grads.size() <= MAX_NUM_INPUTS);

  auto output_grad =
      acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Concat] backward_time = {:.2lf}ms\n",
                 output_grad,
                 input_grads,
                 attrs.axis);
}

TaskImplFunction get_concat_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_concat_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

}; // namespace FlexFlow

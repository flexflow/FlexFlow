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

#include "task-spec/ops/impl/split.h"
#include "kernels/split_kernels.h"
#include "op-attrs/tensor_slot_name.h"
#include "task-spec/profiling.h"
#include "utils/containers/slice.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

static std::vector<TensorSlotName> get_output_slots(SplitAttrs const &attrs) {
  return slice(
      get_variadic_outputs_slot_name_sequence(), 0, attrs.splits.size());
}

static std::vector<GenericTensorAccessorR>
    get_outputs(TaskArgumentAccessor const &acc, SplitAttrs const &attrs) {
  return transform(
      get_output_slots(attrs),
      [&](TensorSlotName output_slot_name) -> GenericTensorAccessorR {
        return acc.get_tensor<Permissions::RO>(output_slot_name);
      });
}

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  SplitAttrs attrs = acc.get_op_attrs().require_split();

  GenericTensorAccessorR input =
      acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);

  std::vector<GenericTensorAccessorW> outputs =
      transform(get_output_slots(attrs),
                [&](TensorSlotName output_slot_name) -> GenericTensorAccessorW {
                  return acc.get_tensor<Permissions::WO>(output_slot_name);
                });

  return profile(split_forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Split] forward_time = {:.2lf}ms\n",
                 attrs,
                 input,
                 outputs);
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  SplitAttrs attrs = acc.get_op_attrs().require_split();

  std::vector<GenericTensorAccessorR> outputs = get_outputs(acc, attrs);

  std::vector<GenericTensorAccessorR> output_grads =
      transform(get_output_slots(attrs),
                [&](TensorSlotName output_slot_name) -> GenericTensorAccessorR {
                  return acc.get_tensor_grad<Permissions::RO>(output_slot_name);
                });

  GenericTensorAccessorR input =
      acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  GenericTensorAccessorW input_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::INPUT);

  return profile(split_backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Split] backward_time = {:.2lf}ms\n",
                 attrs,
                 outputs,
                 output_grads,
                 input,
                 input_grad);
}

TaskImplFunction get_split_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_split_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

}; // namespace FlexFlow

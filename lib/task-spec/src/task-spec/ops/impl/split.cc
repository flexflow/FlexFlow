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
#include "op-attrs/ops/split.h"
#include "task-spec/profiling.h"
#include "utils/hash-utils.h"
#include "utils/nonnegative_int/nonnegative_range.h"

namespace FlexFlow {

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  SplitAttrs attrs = acc.get_op_attrs().require_split();

  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);

  auto outputs =
      transform(split_get_output_slot_names(attrs),
                [&](TensorSlotName slot_name) -> GenericTensorAccessorW {
                  return acc.get_tensor<Permissions::WO>(slot_name);
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
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  SplitAttrs attrs = acc.get_op_attrs().require_split();

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::INPUT);

  auto output_grads =
      transform(split_get_output_slot_names(attrs),
                [&](TensorSlotName slot_name) -> GenericTensorAccessorR {
                  return acc.get_tensor_grad<Permissions::RO>(slot_name);
                });

  return profile(split_backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Split] backward_time = {:.2lf}ms\n",
                 attrs,
                 output_grads,
                 input_grad);
}

TaskImplFunction get_split_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_split_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

}; // namespace FlexFlow

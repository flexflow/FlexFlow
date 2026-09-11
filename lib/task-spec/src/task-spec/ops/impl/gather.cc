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

#include "task-spec/ops/impl/gather.h"
#include "kernels/gather_kernels.h"
#include "op-attrs/ff_ordered/ff_ordered_get_idxs.h"
#include "task-spec/profiling.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include <optional>

namespace FlexFlow {

using namespace FlexFlow::Kernels::Gather;

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto index = acc.get_tensor<Permissions::RO>(TensorSlotName::INDEX);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  device_handle_t handle = acc.get_ff_handle();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  GatherAttrs attrs = acc.get_op_attrs().require_gather();

  ASSERT(get_num_dims(input.shape.dims) == get_num_dims(index.shape.dims));
  ASSERT(get_num_dims(output.shape.dims) == get_num_dims(index.shape.dims));

  for (ff_dim_t i : ff_ordered_get_idxs(input.shape.dims.ff_ordered)) {
    ASSERT(dim_at_idx(index.shape.dims, i) == dim_at_idx(output.shape.dims, i));
    if (i != attrs.dim) {
      ASSERT(dim_at_idx(input.shape.dims, i) ==
             dim_at_idx(index.shape.dims, i));
    }
  }

  std::optional<GatherPerDeviceState> per_device_state =
      init_kernel(kernel_device_type, handle, attrs.dim);
  return DeviceSpecificPerDeviceOpState{
      acc.make_device_specific(per_device_state),
  };
}

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  GatherPerDeviceState per_device_state =
      acc.get_per_device_op_state().require_gather().value();

  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto index = acc.get_tensor<Permissions::RO>(TensorSlotName::INDEX);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Gather] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input,
                 index,
                 output);
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  GatherPerDeviceState per_device_state =
      acc.get_per_device_op_state().require_gather().value();

  auto output_grad =
      acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);
  auto index = acc.get_tensor<Permissions::RO>(TensorSlotName::INDEX);
  auto input_grad = acc.get_tensor_grad<Permissions::WO>(TensorSlotName::INPUT);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Gather] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 output_grad,
                 index,
                 input_grad);
}

TaskImplFunction get_gather_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}

TaskImplFunction get_gather_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_gather_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

}; // namespace FlexFlow

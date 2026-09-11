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

#include "task-spec/ops/impl/reshape.h"
#include "kernels/reshape_kernels.h"
#include "task-spec/profiling.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Reshape;

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  ReshapeAttrs attrs = acc.get_op_attrs().require_reshape();

  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Reshape] forward time = {:.2lf}ms\n",
                 input,
                 output);
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  ReshapeAttrs attrs = acc.get_op_attrs().require_reshape();

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::INPUT);
  auto output_grad =
      acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Reshape] backward time = {:.2lf}ms\n",
                 output_grad,
                 input_grad);
}

TaskImplFunction get_reshape_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_reshape_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

}; // namespace FlexFlow

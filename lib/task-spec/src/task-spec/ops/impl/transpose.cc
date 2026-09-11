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

#include "task-spec/ops/impl/transpose.h"
#include "kernels/transpose_kernels.h"
#include "op-attrs/ops/transpose.h"
#include "task-spec/profiling.h"
#include "utils/integer_conversions.h"

using namespace FlexFlow::Kernels::Transpose;

namespace FlexFlow {

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  TransposeAttrs attrs = acc.get_op_attrs().require_transpose();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Transpose] Forward_time = {:.2lf} [ms]",
                 attrs,
                 input,
                 output);
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  TransposeAttrs attrs = acc.get_op_attrs().require_transpose();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  auto input_grad = acc.get_tensor_grad<Permissions::WO>(TensorSlotName::INPUT);
  auto output_grad =
      acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Transpose] Backward_time = {:.2lf} [ms]",
                 attrs,
                 output_grad,
                 input_grad);
}

TaskImplFunction get_transpose_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_transpose_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

} // namespace FlexFlow

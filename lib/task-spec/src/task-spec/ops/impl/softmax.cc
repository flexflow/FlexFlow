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

#include "task-spec/ops/impl/softmax.h"
#include "kernels/softmax_kernels.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "task-spec/profiling.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Softmax;

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  device_handle_t handle = acc.get_ff_handle();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  SoftmaxAttrs attrs = acc.get_op_attrs().require_softmax();

  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  positive_int output_w = dim_at_idx(output.shape.dims, legion_dim_t{0_n});
  positive_int output_h = dim_at_idx(output.shape.dims, legion_dim_t{1_n});
  positive_int output_c = dim_at_idx(output.shape.dims, legion_dim_t{2_n});
  positive_int output_n = dim_at_idx(output.shape.dims, legion_dim_t{3_n});

  std::optional<SoftmaxPerDeviceState> per_device_state =
      init_kernel(kernel_device_type,
                  handle,
                  attrs.dim,
                  output_n.int_from_positive_int(),
                  output_c.int_from_positive_int(),
                  output_h.int_from_positive_int(),
                  output_w.int_from_positive_int());

  return DeviceSpecificPerDeviceOpState{
      acc.make_device_specific(per_device_state),
  };
}

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  SoftmaxPerDeviceState per_device_state =
      acc.get_per_device_op_state().require_softmax().value();

  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Softmax] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr());
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  SoftmaxPerDeviceState per_device_state =
      acc.get_per_device_op_state().require_softmax().value();

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::INPUT);
  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  assert(input_grad.shape == input.shape);

  auto output_grad =
      acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);
  auto output = acc.get_tensor<Permissions::RO>(TensorSlotName::OUTPUT);

  assert(output_grad.shape == output.shape);

  return profile(
      backward_kernel,
      profiling,
      kernel_device_type,
      "[Softmax] backward_time = {:.2lf}ms\n",
      output_grad.get_float_ptr(),
      input_grad.get_float_ptr(),
      get_num_elements(output_grad.shape.dims).int_from_positive_int());
}

TaskImplFunction get_softmax_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}

TaskImplFunction get_softmax_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_softmax_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

}; // namespace FlexFlow

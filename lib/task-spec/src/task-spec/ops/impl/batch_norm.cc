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

#include "task-spec/ops/impl/batch_norm.h"
#include "kernels/batch_norm_kernels.h"
#include "task-spec/profiling.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::BatchNorm;

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  Allocator allocator = acc.get_allocator();
  device_handle_t handle = acc.get_ff_handle();
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);
  BatchNormAttrs attrs = acc.get_op_attrs().require_batch_norm();

  positive_int output_w = dim_at_idx(output.shape.dims, legion_dim_t{0_n});
  positive_int output_h = dim_at_idx(output.shape.dims, legion_dim_t{1_n});
  positive_int output_c = dim_at_idx(output.shape.dims, legion_dim_t{2_n});
  positive_int output_n = dim_at_idx(output.shape.dims, legion_dim_t{3_n});

  float *runningMean;

  std::optional<BatchNormPerDeviceState> per_device_state = init_kernel(
      /*device_type=*/kernel_device_type,
      /*handle=*/handle,
      /*allocator=*/allocator,
      /*runningMean=*/runningMean,
      /*output_n=*/output_n.int_from_positive_int(),
      /*output_c=*/output_c.int_from_positive_int(),
      /*output_h=*/output_h.int_from_positive_int(),
      /*output_w=*/output_w.int_from_positive_int(),
      /*relu=*/attrs.relu);

  return DeviceSpecificPerDeviceOpState{
      acc.make_device_specific(per_device_state),
  };
}

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_per_device_op_state().require_batch_norm().value();
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);
  auto scale = acc.get_tensor<Permissions::RO>(TensorSlotName::SCALE);
  auto bias = acc.get_tensor<Permissions::RO>(TensorSlotName::BIAS);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[BatchNorm] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr(),
                 scale.get_float_ptr(),
                 bias.get_float_ptr());
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  BatchNormPerDeviceState per_device_state =
      acc.get_per_device_op_state().require_batch_norm().value();
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto input_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::INPUT);
  auto output = acc.get_tensor<Permissions::RO>(TensorSlotName::OUTPUT);
  auto output_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::OUTPUT);
  auto scale = acc.get_tensor<Permissions::RO>(TensorSlotName::SCALE);
  auto scale_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::SCALE);
  auto bias_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::BIAS);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[BatchNorm] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 output.get_float_ptr(),
                 output_grad.get_float_ptr(),
                 input.get_float_ptr(),
                 input_grad.get_float_ptr(),
                 scale.get_float_ptr(),
                 scale_grad.get_float_ptr(),
                 bias_grad.get_float_ptr(),
                 get_num_elements(output.shape.dims).int_from_positive_int());
}

TaskImplFunction get_batch_norm_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}

TaskImplFunction get_batch_norm_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_batch_norm_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

}; // namespace FlexFlow

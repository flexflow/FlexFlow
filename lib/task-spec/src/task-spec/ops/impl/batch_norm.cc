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

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  Allocator allocator = acc.get_allocator();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  BatchNormAttrs attrs = acc.get_op_attrs().require_batch_norm();

  TensorShape input_shape = acc.get_tensor_shape(TensorSlotName::INPUT);
  TensorShape output_shape = acc.get_tensor_shape(TensorSlotName::OUTPUT);

  std::optional<BatchNormPerDeviceState> per_device_state =
      batch_norm_init_kernel(
          kernel_device_type, allocator, attrs, input_shape, output_shape);

  return DeviceSpecificPerDeviceOpState{
      acc.make_device_specific(per_device_state),
  };
}

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  device_handle_t handle = acc.get_ff_handle();
  std::optional<BatchNormPerDeviceState> per_device_state =
      acc.get_per_device_op_state().require_batch_norm();
  BatchNormAttrs attrs = acc.get_op_attrs().require_batch_norm();

  GenericTensorAccessorR input =
      acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  GenericTensorAccessorR gamma =
      acc.get_tensor<Permissions::RO>(TensorSlotName::GAMMA);
  GenericTensorAccessorR beta =
      acc.get_tensor<Permissions::RO>(TensorSlotName::BETA);
  GenericTensorAccessorW output =
      acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  return profile(batch_norm_forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[BatchNorm] forward_time = {:.2lf}ms\n",
                 handle,
                 per_device_state,
                 attrs,
                 input,
                 gamma,
                 beta,
                 output);
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  device_handle_t handle = acc.get_ff_handle();
  std::optional<BatchNormPerDeviceState> per_device_state =
      acc.get_per_device_op_state().require_batch_norm();
  BatchNormAttrs attrs = acc.get_op_attrs().require_batch_norm();

  GenericTensorAccessorR input =
      acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  GenericTensorAccessorW input_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::INPUT);
  GenericTensorAccessorR output =
      acc.get_tensor<Permissions::RO>(TensorSlotName::OUTPUT);
  GenericTensorAccessorR output_grad =
      acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);
  GenericTensorAccessorR gamma =
      acc.get_tensor<Permissions::RO>(TensorSlotName::GAMMA);
  GenericTensorAccessorW gamma_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::GAMMA);
  GenericTensorAccessorW beta_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::BETA);

  return profile(batch_norm_backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[BatchNorm] backward_time = {:.2lf}ms\n",
                 handle,
                 per_device_state,
                 attrs,
                 output,
                 output_grad,
                 input,
                 input_grad,
                 gamma,
                 gamma_grad,
                 beta_grad);
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
